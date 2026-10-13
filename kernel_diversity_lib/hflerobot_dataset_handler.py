# hf_lerobot_stack.py
from typing import List, Dict, Any, Sequence
from collections import defaultdict
import numpy as np
from datasets import load_dataset

from rich import print as rprint

if __package__:
    from .embeddings import ImageEmbedder
else:
    from embeddings import ImageEmbedder


# Default numeric and image features; callers can override either list.
NUMERIC_KEYS: List[str] = [
    "observation.state",
    "action",
    "next.success",
]

IMAGE_KEYS: List[str] = [
    "observation.image",
]

TIME_STRIDE = 1           # sample every #TIME_STRIDE time/horizon steps
INCLUDE_LAST = True       # also keep the final frame even if not on the grid


def _get_by_dotted_key(record, dotted: str):
    # 1) Fast path for flat, parquet-style columns: "observation.images.top"
    if dotted in record:
        return record[dotted]

    # 2) Fallback: walk nested dicts: record["observation"]["images"]["top"]
    node = record
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        if isinstance(node, dict) and part in node:
            node = node[part]
            continue
        # Last-chance: sometimes only some levels are flat; try remaining tail as a flat key
        tail = ".".join(parts[i:])
        if tail in record:
            return record[tail]
        raise KeyError(parts[i])
    return node



def _as_1d_array(x: Any) -> np.ndarray:
    """Coerce scalars / lists / tensors to 1D numpy arrays."""
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32).reshape(-1)
    # booleans or scalars
    if isinstance(x, (bool, int, float, np.floating, np.integer, np.bool_)):
        return np.asarray([x], dtype=np.float32)
    # HF can return numpy scalars / arrays already
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).reshape(-1)
    # Fallback: try to convert
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _as_uint8_image(img_obj: Any) -> np.ndarray:
    """
    Convert HF Image feature (PIL or numpy) to HxWxC uint8.
    Datasets with Image() features usually yield PIL.Image or np.uint8 arrays.
    """
    try:
        import PIL.Image  # noqa
        if hasattr(img_obj, "convert"):
            return np.array(img_obj.convert("RGB"), dtype=np.uint8)
    except Exception:
        pass
    arr = np.array(img_obj)
    if arr.dtype != np.uint8:
        # Normalize / cast conservatively; tweak as needed.
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


class HFLeRobotReducer:
    """
    Loads a LeRobot/HF dataset, groups by episode, builds a long-vector per time step
    from user-specified keys, embeds images using ImageEmbedder, and returns a padded
    array of shape (N, T_max, d) where shorter episodes repeat their last step.
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        numeric_keys: Sequence[str] = None,
        image_keys: Sequence[str] = None,
        embedding: str = "clip",
        embedding_output_dim: int = None,
        embedder_batch_size: int = None,
        hf_load_kwargs: dict = None,
        time_stride: int = TIME_STRIDE,
        episodes_filter: Sequence[int] | None = None,
        include_last: bool = INCLUDE_LAST,
    ):
        self.repo_id = repo_id
        self.split = split
        if not isinstance(time_stride, (int, np.integer)) or time_stride < 1:
            raise ValueError("time_stride must be a positive integer")
        self.time_stride = time_stride
        self.include_last = include_last
        self.episodes_filter = list(episodes_filter or [])
        self._episodes_filter_set = set(self.episodes_filter)
        if isinstance(numeric_keys, (str, bytes)) or isinstance(image_keys, (str, bytes)):
            raise ValueError("Numeric and image keys must be sequences of nonblank strings")
        self.numeric_keys = list(NUMERIC_KEYS if numeric_keys is None else numeric_keys)
        self.image_keys = list(IMAGE_KEYS if image_keys is None else image_keys)
        modality_names = self.numeric_keys + self.image_keys
        if not modality_names:
            raise ValueError("At least one numeric or image key is required")
        if any(not isinstance(key, str) or not key.strip() for key in modality_names):
            raise ValueError("Numeric and image modality keys must be nonblank strings")
        if len(set(modality_names)) != len(modality_names):
            raise ValueError("Numeric and image modality keys must be unique")
        self.hf_load_kwargs = hf_load_kwargs or {}

        # Load dataset (arrow table). You can also use streaming if needed.
        # NOTE: the repo exposes per-frame rows with 'episode_index' and 'frame_index'.
        self.ds = load_dataset(self.repo_id, split=self.split, **self.hf_load_kwargs)

        # Image embedder (uses your ImageEmbedder implementation)
        self.embedder = None
        if self.image_keys:
            self.embedder = ImageEmbedder(
                embedding=embedding,
                output_dim=embedding_output_dim,
                batch_size=embedder_batch_size,
            )

        self.index_to_episode_id = []   # maps array row -> episode id
        self.n_to_N = []
        self.global_episode_ids = []
        self.data_array = None          # (N, T_max, d)
        self.modality_slices = {}        # modality name -> feature-column slice
        self._build()

    def _build(self):
        # Build global episode ordering without materializing frames
        all_episode_ids = self.ds.unique("episode_index")  # list of episode ids
        all_episode_ids = [int(x) for x in all_episode_ids]
        all_episode_ids_sorted = sorted(all_episode_ids)
        self.global_episode_ids = all_episode_ids_sorted  # full global order
        ep_to_global_idx = {ep_id: i for i, ep_id in enumerate(all_episode_ids_sorted)}

        ds_sel = self.ds
        if self._episodes_filter_set:
            allowed = np.array(sorted(self._episodes_filter_set), dtype=np.int64)
            # Vectorized batched filter to avoid Python-per-row overhead
            ds_sel = self.ds.filter(
                lambda batch, allowed=allowed: np.isin(np.asarray(batch["episode_index"]), allowed),
                batched=True
            )

        # 1) Group rows by episode_index
        by_ep = defaultdict(list)
        for row in ds_sel:
            by_ep[int(row["episode_index"])].append(row)

        # 2) Sort only the kept episodes
        for ep in by_ep:
            by_ep[ep].sort(key=lambda r: int(r["frame_index"]))

        # --- establish global episode order & init maps ---
        all_eps_sorted = sorted(by_ep.items(), key=lambda kv: kv[0])
        self.index_to_episode_id = []
        self.n_to_N = []

        # 3) Build sequences (T, d) per episode
        flat_sequences = []
        max_T = 0
        expected_layout = None
        self.modality_slices = {}

        for ep_id, frames in all_eps_sorted:
            # Skip if filtering is active and this episode isn't requested
            if self._episodes_filter_set and (ep_id not in self._episodes_filter_set):
                continue
            # maintain maps for selected episodes
            self.index_to_episode_id.append(ep_id)  # n -> ep_id
            self.n_to_N.append(ep_to_global_idx[ep_id])  # n -> N

            n_idx = len(self.index_to_episode_id)  # next n-row to be filled
            global_idx = ep_to_global_idx[ep_id]
            rprint(f"Computing embedding for [green]{global_idx}[/green]...")

            T = len(frames)

            # 3a) Prepare image embeddings per key (batched for speed)
            img_embeds_by_key = {}
            for k in self.image_keys:
                # Collect the episode’s images for this key
                imgs = [_as_uint8_image(_get_by_dotted_key(fr, k)) for fr in frames]
                if self.embedder is None:
                    raise ValueError("Image keys provided but embedder is None.")
                # (T, D_img)
                emb = self.embedder.embed(imgs).numpy()
                if emb.ndim == 0 or emb.shape[0] != T:
                    raise ValueError(f"Episode {ep_id} image modality {k!r} has an inconsistent time dimension")
                emb = emb.reshape(T, -1)
                img_embeds_by_key[k] = emb

            # 3b) Build the long vector per time step by concatenating
            # subsample indices for this episode
            T_full = len(frames)
            indices = list(range(0, T_full, self.time_stride))
            if self.include_last and indices and indices[-1] != T_full - 1:
                indices.append(T_full - 1)

            per_t_vectors = []
            for i in indices:  # <-- now only 0, 9, 18, ...
                fr = frames[i]
                chunks = []

                # numeric keys at timestep i
                for k in self.numeric_keys:
                    val = _get_by_dotted_key(fr, k)
                    vec = _as_1d_array(val)
                    chunks.append(vec)

                # image embeddings at timestep i
                # (since you embedded ALL frames above, index with i)
                for k in self.image_keys:
                    chunks.append(img_embeds_by_key[k][i])

                layout = [(name, chunk.size)
                          for name, chunk in zip(self.numeric_keys + self.image_keys, chunks)]
                if any(width == 0 for _, width in layout):
                    raise ValueError(f"Episode {ep_id} frame {i} contains an empty modality")
                if expected_layout is None:
                    expected_layout = layout
                    start = 0
                    for name, width in layout:
                        self.modality_slices[name] = slice(start, start + width)
                        start += width
                elif layout != expected_layout:
                    raise ValueError(f"Episode {ep_id} frame {i} has inconsistent modality feature dimensions")

                long_vec = np.concatenate(chunks, axis=0).astype(np.float32)
                per_t_vectors.append(long_vec)

            seq = np.stack(per_t_vectors, axis=0)  # (T_subsampled, d)
            flat_sequences.append(seq)
            #max_T = max(max_T, T)
            T_sub = seq.shape[0]
            max_T = max(max_T, T_sub)

        # 4) Pad to (N, T_max, d) by repeating the last element
        N = len(flat_sequences)
        if N == 0:
            raise ValueError("No episodes remain after applying the episode filter")
        d = flat_sequences[0].shape[1]
        arr = np.zeros((N, max_T, d), dtype=np.float32)

        for i, seq in enumerate(flat_sequences):
            """T = seq.shape[0]
            arr[i, :T] = seq
            if T < max_T:
                arr[i, T:] = seq[-1]  # repeat last step"""
            T_sub = seq.shape[0]
            arr[i, :T_sub] = seq
            if T_sub < max_T:
                arr[i, T_sub:] = seq[-1]  # repeat last step

        self.data_array = arr

    def get_demos(self) -> np.ndarray:
        """Return the (N, T_max, d) array."""
        return self.data_array

    def get_modalities(self) -> Dict[str, np.ndarray]:
        """Return named numeric-field and image-embedding trajectory views."""
        return {name: self.data_array[..., columns]
                for name, columns in self.modality_slices.items()}

    def get_index_to_episode(self) -> List[int]:
        """Map row index to the source episode_index."""
        return self.index_to_episode_id

    def get_diversity_score(self, metric) -> float:
        return metric(self.data_array)

    def map_back(self, n_indices: Sequence[int]) -> List[int]:
        """
        Given a list of indices in [0..n-1] (rows in the returned numpy array),
        return the corresponding global indices [0..N-1] (order among *all* episodes).
        """
        # sanity checks are optional but helpful
        n_total = len(self.n_to_N)
        out = []
        for i in n_indices:
            if i < 0 or i >= n_total:
                raise IndexError(f"n index {i} out of range [0, {n_total - 1}]")
            out.append(self.n_to_N[i])
        return out
