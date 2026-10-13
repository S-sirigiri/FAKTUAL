from email.policy import default

import h5py
import os
from typing import Sequence

from rich import print as rprint

import numpy as np
from sklearn.preprocessing import (
    normalize,
    MinMaxScaler,
    StandardScaler,
    RobustScaler
)

if __package__:
    from .embeddings import ImageEmbedder
else:
    from embeddings import ImageEmbedder

TIME_STRIDE = 1           # sample every #TIME_STRIDE time/horizon steps
INCLUDE_LAST = True       # also keep the final frame even if not on the grid

DEFAULT_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "agentview_image",
    "robot0_eye_in_hand_image"
]

class HDF5DatasetReducer:
    def __init__(
        self,
        input_path: str,
        embedding: str = "clip",
        embedding_dim: int = None,
        time_stride: int = TIME_STRIDE,
        keys: Sequence[str] = None,
        include_last: bool = INCLUDE_LAST,
    ):
        """
        Initializes the reducer with the path to the input HDF5 file.
        The output path will automatically be <input_basename>_reduced.hdf5.
        """
        self.input_path = input_path
        self.train_ids = []
        self.data_array = None    # Will hold the (N, T, d) array
        self.modality_slices = {} # Maps modality names to feature-column slices
        self.index_to_id = {}     # Maps array index -> demo_id
        if not isinstance(time_stride, (int, np.integer)) or time_stride < 1:
            raise ValueError("time_stride must be a positive integer")
        self.time_stride = time_stride
        self.include_last = include_last
        if isinstance(keys, (str, bytes)):
            raise ValueError("keys must be a sequence of nonblank strings")
        self._explicit_keys = keys is not None
        self.keys = list(DEFAULT_KEYS if keys is None else keys)
        if not self.keys:
            raise ValueError("At least one observation key is required")
        if any(not isinstance(key, str) or not key.strip() for key in self.keys):
            raise ValueError("Observation keys must be nonblank strings")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Observation keys must be unique")

        # Initialize the image embedder for any uint8 image fields
        if embedding is not None:
            if embedding_dim is None:
                self.embedder = ImageEmbedder(embedding=embedding)
            else:
                self.embedder = ImageEmbedder(embedding=embedding, output_dim=embedding_dim)
        else:
            self.embedder = None

        self._load_train_ids()
        self._build_padded_array()

    def _load_train_ids(self):
        """Read the train mask from the input file."""
        with h5py.File(self.input_path, 'r') as f:
            raw = f['mask']['train'][()]
            # Convert byte-strings to Python str
            self.train_ids = [s.decode('ascii') for s in raw]

    def _build_padded_array(self):
        """
        Loads each train demo, flattens per-timestep features into a vector,
        and pads all to the maximum trajectory length.
        """
        flat_sequences = []
        max_T = 0
        expected_layout = None
        self.modality_slices = {}

        with h5py.File(self.input_path, 'r') as f:
            # Check all explicit selections before embedding any episode images.
            if self._explicit_keys:
                for demo_id in self.train_ids:
                    for group_name in ('obs', 'next_obs'):
                        group = f['data'][demo_id][group_name]
                        missing = [key for key in self.keys if key not in group]
                        if missing:
                            raise ValueError(
                                f"Demo {demo_id!r} group {group_name!r} is missing selected keys: {missing}"
                            )

            for idx, demo_id in enumerate(self.train_ids):
                grp = f['data'][demo_id]

                # --- compute subsampling indices once per episode ---
                # Use actions length (or dones) as the authoritative T
                T_full = grp['actions'].shape[0]
                if T_full == 0:
                    raise ValueError(f"Demo {demo_id!r} has no trajectory steps")
                indices = list(range(0, T_full, self.time_stride))
                if self.include_last and T_full > 0 and (not indices or indices[-1] != T_full - 1):
                    indices.append(T_full - 1)

                # Flatten all obs fields, embedding images when encountered
                obs_grp = grp['obs']
                obs_list = []
                modality_parts = []
                print_flag = True
                for k in sorted(obs_grp.keys()):
                    if k not in self.keys:
                        continue
                    data = obs_grp[k][()]
                    if data.ndim == 0 or data.shape[0] != T_full:
                        raise ValueError(f"Demo {demo_id!r} field 'obs.{k}' has an inconsistent time dimension")
                    data = data[indices] # <--- SUBSAMPLE TIME
                    # Embed uint8 image sequences
                    if data.dtype == np.uint8 and data.ndim == 4:
                        if self.embedder is not None:
                            if print_flag: rprint(f"Computing embedding for [green]{demo_id}[/green]...")
                            print_flag = False
                            emb = self.embedder.embed(data)          # (T_sub, D_img)
                            emb = emb.numpy()
                            if emb.ndim == 0 or emb.shape[0] != len(indices):
                                raise ValueError(f"Demo {demo_id!r} image modality 'obs.{k}' has an inconsistent time dimension")
                            emb = emb.reshape(len(indices), -1)
                            #emb = MinMaxScaler().fit_transform(emb)
                            obs_list.append(emb)
                            modality_parts.append((f"obs.{k}", emb))
                    else:
                        arr = data.reshape(len(indices), -1)
                        #arr = MinMaxScaler().fit_transform(arr)
                        obs_list.append(arr)
                        modality_parts.append((f"obs.{k}", arr))
                if not obs_list:
                    raise ValueError(f"Demo {demo_id!r} has no selected observation features")
                obs = np.concatenate(obs_list, axis=1)

                # Flatten all next_obs fields similarly
                nxt_grp = grp['next_obs']
                nxt_list = []
                for k in sorted(nxt_grp.keys()):
                    if k not in self.keys:
                        continue
                    data = nxt_grp[k][()]
                    if data.ndim == 0 or data.shape[0] != T_full:
                        raise ValueError(f"Demo {demo_id!r} field 'next_obs.{k}' has an inconsistent time dimension")
                    data = data[indices]  # <--- SUBSAMPLE TIME
                    if data.dtype == np.uint8 and data.ndim == 4:
                        if self.embedder is not None:
                            emb = self.embedder.embed(data)
                            emb = emb.numpy()
                            if emb.ndim == 0 or emb.shape[0] != len(indices):
                                raise ValueError(f"Demo {demo_id!r} image modality 'next_obs.{k}' has an inconsistent time dimension")
                            emb = emb.reshape(len(indices), -1)
                            #emb = MinMaxScaler().fit_transform(emb)
                            nxt_list.append(emb)
                            modality_parts.append((f"next_obs.{k}", emb))
                    else:
                        arr = data.reshape(len(indices), -1)
                        #arr = MinMaxScaler().fit_transform(arr)
                        nxt_list.append(arr)
                        modality_parts.append((f"next_obs.{k}", arr))
                if not nxt_list:
                    raise ValueError(f"Demo {demo_id!r} has no selected next-observation features")
                next_obs = np.concatenate(nxt_list, axis=1)

                for name in ('dones', 'rewards', 'states'):
                    if grp[name].ndim == 0 or grp[name].shape[0] != T_full:
                        raise ValueError(f"Demo {demo_id!r} field {name!r} has an inconsistent time dimension")

                actions = grp['actions'][()][indices].reshape(len(indices), -1)
                #actions = MinMaxScaler().fit_transform(actions)

                dones   = grp['dones'][()][indices].reshape(len(indices), -1)
                #dones = MinMaxScaler().fit_transform(dones)

                rewards = grp['rewards'][()][indices].reshape(len(indices), -1)
                #rewards = MinMaxScaler().fit_transform(rewards)

                states  = grp['states'][()][indices].reshape(len(indices), -1)
                #states = MinMaxScaler().fit_transform(states)

                # Concat into shape (T, d)
                fields = [obs, next_obs, actions, dones, rewards, states]
                seq = np.concatenate(fields, axis=1)
                modality_parts.extend([
                    ("actions", actions), ("dones", dones),
                    ("rewards", rewards), ("states", states),
                ])
                layout = [(name, part.shape[1]) for name, part in modality_parts]
                if any(width == 0 for _, width in layout):
                    raise ValueError(f"Demo {demo_id!r} contains an empty modality")
                if expected_layout is None:
                    expected_layout = layout
                    start = 0
                    for name, width in layout:
                        self.modality_slices[name] = slice(start, start + width)
                        start += width
                elif layout != expected_layout:
                    raise ValueError(f"Demo {demo_id!r} has inconsistent modality names or feature dimensions")
                flat_sequences.append(seq)

                max_T = max(max_T, seq.shape[0])
                self.index_to_id[idx] = demo_id

        N = len(flat_sequences)
        if N == 0:
            raise ValueError("The dataset train mask contains no demos")
        d = flat_sequences[0].shape[1]

        # Pad to shape (N, max_T, d)
        arr = np.zeros((N, max_T, d), dtype=flat_sequences[0].dtype)
        for idx, seq in enumerate(flat_sequences):
            T = seq.shape[0]
            arr[idx, :T, :] = seq
            if T < max_T:
                arr[idx, T:, :] = seq[-1]  # repeat last frame

        self.data_array = arr

    def get_demos(self)-> np.ndarray:
        #self.data_array = np.random.permutation(self.data_array)
        return self.data_array

    def get_modalities(self):
        """Return named trajectory views for each observation and numeric field."""
        return {name: self.data_array[..., columns]
                for name, columns in self.modality_slices.items()}

    def get_diversity_score(self, metric) -> float:
        return metric(self.data_array)

    def reduce(self, top_idxes, metric=None) -> np.ndarray:
        """
        """
        rprint('The dataset ' + "'" + self.input_path + "'" + ' has ' + str(len(self.train_ids)) + ' train demos.')
        if metric is not None:
            rprint(f"Score of original dataset: {metric(self.data_array):.{16}f}")

        reduced_data = self.data_array[list(top_idxes)]

        if metric is not None:
            rprint(f"Score of reduced dataset: {metric(reduced_data):.{16}f}")

        return reduced_data

    def process(self, top_idxes, output_path, output_suffix, metric=None):
        """
        Runs reduce(), then writes out a new HDF5 (same name + "_reduced"),
        copying everything except updating only mask/train.
        """
        reduced = self.reduce(top_idxes, metric=metric)
        M = reduced.shape[0]
        #kept_ids = [self.index_to_id[i] for i in range(M)]
        kept_ids = [self.index_to_id[i] for i in top_idxes]
        return self._write_new_file(kept_ids, output_path, output_suffix)

    def _write_new_file(self, kept_ids, output_path, output_suffix):
        """
        Copy the entire original file to a new one (appending "_@output_suffix"
        to the basename), but overwrite only mask/train.
        """
        base, ext = os.path.splitext(self.input_path)
        filename = os.path.basename(base)
        if output_path is None:
            out_path = f"{base}_{output_suffix}{ext}"
        else:
            out_path = output_path
            out_path = os.path.join(output_path, f"{filename}_{output_suffix}{ext}")

        with h5py.File(self.input_path, 'r') as src, h5py.File(out_path, 'w') as dst:
            # 1) Copy all top-level groups and datasets (including attrs)
            for name in src:
                if name != 'mask':
                    src.copy(name, dst)

            # 2) Copy root attributes
            for attr_key, attr_val in src.attrs.items():
                dst.attrs[attr_key] = attr_val

            # 3) Rebuild mask/, copying everything except train
            mask_grp = dst.create_group('mask')
            for ds_name in src['mask']:
                if ds_name == 'train':
                    continue
                src['mask'].copy(ds_name, mask_grp)

            # 4) Write updated train mask
            dt = h5py.string_dtype('ascii')
            encoded = [s.encode('ascii') for s in kept_ids]
            mask_grp.create_dataset('train', data=encoded, dtype=dt)

        rprint(f"New dataset written to '{out_path}' with {len(kept_ids)} train demos.")

        return out_path
