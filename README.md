# FAKTUAL: Diversity for Robot Demonstration Datasets

Code accompanying **Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets**.

*Sreevardhan Sirigiri, Nathan Samuel de Lara, Christopher Agia, Florian Shkurti, and Fabio Ramos*

**FAKTUAL** (FAst trajectory Kernel enTropy cUration for imitation Learning) measures dataset diversity and curates subsets of robot demonstrations using trajectory kernels. Its selection criterion depends on the dataset itself, without requiring a trained imitation policy, an auxiliary scoring model, or policy rollouts. Images can be represented using pretrained visual encoders.

This README provides a general guide to `kernel_diversity_lib`.

## How it works

Robot demonstrations contain sequences of states, actions, images, and other observations. The library represents each demonstration as a path, compares paths using a sequence kernel, and measures diversity from the resulting similarity matrix.

For a normalized kernel matrix $K$ over $N$ demonstrations, entropy is computed from the eigenvalues $\lambda_i$ of $K/N$:

$$
H = -\sum_i \lambda_i \log \lambda_i.
$$

Shannon and von Neumann entropy coincide in this setting. Exponentiating entropy gives the **Vendi score**, which expresses the effective number of distinct demonstrations under the chosen kernel. Determinant-based objectives provide another way to measure diversity.

The library supports two main workflows:

- **Measure diversity:** compute a score for a demonstration dataset.
- **Curate a subset:** select demonstrations under a size budget using a diversity objective, then use that selection in a downstream learning pipeline.

Diversity complements demonstration quality and sufficient coverage around useful behaviors. A higher diversity score alone does not guarantee better policy performance.

## Library structure

| Module | Purpose |
| --- | --- |
| `dataset_reducer.py` | Command-line interface for scoring and subset selection |
| `HDF5_handler.py` | Loading and selection output for RoboMimic HDF5 datasets |
| `hflerobot_dataset_handler.py` | Loading Hugging Face / LeRobot-style demonstration datasets |
| `embeddings.py` | CLIP and DINOv2 image features |
| `kernels.py` | Sequence kernels and combinations of modality-specific kernels |
| `diversity_metrics.py` | Shannon entropy, von Neumann entropy, Vendi score, and determinant objectives |
| `maximizer.py` | Subset selection algorithms |

The rest of the repository contains policy training code, configurations, baselines, and analysis utilities used in the research.

## Setup

The root-level `lerobot/` directory is a Git submodule pointing to [Hugging Face LeRobot](https://github.com/huggingface/lerobot). Initialize it after cloning:

```bash
git submodule update --init --recursive
```

The library requires Python **3.10 or later**. Its current kernel and metric pipeline uses CuPy and requires a compatible NVIDIA GPU/CUDA environment. The supplied requirements select CuPy for **CUDA 12**; assigning the Torch signature solver to the CPU does not make the full pipeline CPU-only.

From the repository root, install NumPy, scikit-learn, and a CUDA-compatible PyTorch build, followed by the library requirements:

```bash
python -m pip install numpy scikit-learn torch
python -m pip install -r kernel_diversity_lib/requirements.txt
```

The requirements include the project's SigKernel and KSig forks, installed from GitHub. Git and a compiler toolchain may be needed to build them. CLIP and DINOv2 weights are downloaded on first use.

Policy training has additional dependencies. See the bundled [robomimic documentation](robomimic/README.md) or the relevant baseline documentation for those environments.

## General workflow

1. **Choose the candidate demonstrations.** Prepare a dataset in one of the supported formats and identify the training demonstrations to consider.
2. **Choose the representation.** Select observation fields and, when using images, an embedding model. Temporal subsampling can reduce the length of the paths used for comparison.
3. **Choose a kernel and objective.** Start with the default random Fourier signature features kernel and an entropy-based metric, then adjust settings for your dataset.
4. **Score or select.** Compute the dataset's diversity, or specify a subset size and selection strategy.
5. **Use the selection downstream.** Train on the selected demonstrations and evaluate whether the chosen budget and representation suit your task.

Run commands from the repository root. The complete interface is available through:

```bash
python -m kernel_diversity_lib.dataset_reducer --help
```

### RoboMimic HDF5 datasets

The adapter reads candidate demonstrations from `mask/train`. Each referenced group under `data/` must contain `obs`, `next_obs`, `actions`, `dones`, `rewards`, and `states`.

By default, observation features include end-effector position and orientation, gripper position, and the `agentview_image` and `robot0_eye_in_hand_image` camera fields when available. Use `--keys` to choose observation fields from both `obs` and `next_obs`. Actions, terminal flags, rewards, and simulator states are also included in the path representation.

To measure diversity, replace the input path below with your dataset:

```bash
python -m kernel_diversity_lib.dataset_reducer \
  --input /path/to/dataset.hdf5 \
  --get-score \
  --metric shannon
```

To select a subset, provide a positive budget no larger than the number of candidate demonstrations. For example:

```bash
python -m kernel_diversity_lib.dataset_reducer \
  --input /path/to/dataset.hdf5 \
  --maximizer submodular \
  --stochastic-greedy \
  --metric shannon \
  --k 50 \
  --output-suffix diverse
```

The output is `/path/to/dataset_diverse.hdf5`. It copies the source file and replaces **only `mask/train`** with the selected demonstration IDs. Other masks and all stored demonstrations remain present. Your downstream loader must honor the training mask; the output is not a physically smaller dataset. For HDF5 inputs, `--output-path` can specify an existing destination directory.

### Hugging Face / LeRobot-style datasets

Use `--dataset-type lerobot` and provide a dataset ID compatible with Hugging Face Datasets. The adapter expects per-frame rows with `episode_index` and `frame_index`, plus your selected numeric and image fields.

The defaults are numeric fields `observation.state`, `action`, and `next.success`, and image field `observation.image`. Override them to match your dataset. Image fields must yield decoded PIL or NumPy images; the adapter does not decode video references itself.

For example, select from state and action trajectories without image embeddings:

```bash
python -m kernel_diversity_lib.dataset_reducer \
  --dataset-type lerobot \
  --input organization/dataset \
  --numeric-keys observation.state action \
  --image-keys '[]' \
  --maximizer submodular \
  --stochastic-greedy \
  --metric shannon \
  --k 50 \
  --output-path selected_episodes.json
```

Replace `organization/dataset` and the field names with your own. Use `--image-keys` to include image columns, or `--episode-list` to restrict the candidate source episode IDs.

The output JSON contains **positions in the sorted list of all source episode IDs**, rather than a dataset copy. These positions equal the original `episode_index` values only when those IDs are contiguous and start at zero. An existing output JSON is left unchanged and the command exits.

## Choosing settings

| Setting | Guidance |
| --- | --- |
| Diversity metric | `shannon` or `von_neumann` for entropy; `vendi` for effective diversity; `det` or `logdet` for determinant-based objectives. |
| Sequence kernel | The default is `"Random fourier signature features kernel"`. Full signature, random warping series, and global alignment kernels are also available. |
| Signature approximation | `--n-levels`, `--n-components`, and `--method` control the random Fourier signature representation. Defaults are `5`, `100`, and `TRP`. |
| Kernel bandwidth | Estimated from the dataset unless a value is supplied with `--bandwidth`. Keep the representation and kernel settings consistent when comparing scores. |
| Image representation | `--embedding clip` (default) or `--embedding dinov2`. Leaving `--embedding-dim` unset uses the encoder's native feature dimension. |
| Temporal sampling | `--time-stride` selects every Nth timestep. The final timestep is retained unless `--no-include-last` is set. Saved HDF5 trajectories retain their original resolution. |
| Multimodal combination | `--combination-method concatenate` combines features into one path; `convex` combines normalized modality-specific kernels with nonnegative weights. |
| Selection | Stochastic greedy selection is illustrated above. Other strategies include black-box search, DPP sampling, and a random baseline. Use `--maximizer random --random-state 1` for a seeded random comparison. |
| Score after selection | Add `--get-score-after-maximizing` to report diversity before and after selection. |

Both adapters pad shorter feature sequences by repeating the final timestep. With convex combination, weights default to uniform; `--modality-weights` accepts a JSON dictionary for explicit weights, and `--modality-kernel-configs` accepts per-modality kernel settings.

Some implementation details matter when choosing other options: `--embedding-dim` uses a randomly initialized linear projection, and `--random-state` does not seed every randomized selector. The current `blackbox` greedy initialization takes candidates in input order; positive local-search iterations can improve the selection, while `--cross-entropy` uses a separate search. The `mixed-objective` path currently uses that initialization without local search and should not be treated as a reproduction of the paper's hybrid selection method.

## Citation

If you use this work, please cite:

```bibtex
@misc{sirigiri2026diversityactuallymeasurefast,
      title={Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets}, 
      author={Sreevardhan Sirigiri and Nathan Samuel de Lara and Christopher Agia and Florian Shkurti and Fabio Ramos},
      year={2026},
      eprint={2603.11634},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.11634}, 
}
```

Please also cite the underlying frameworks and baselines when using them. Vendored projects retain their own licenses: [robomimic](robomimic/LICENSE), [CUPID](baselines/cupid/LICENSE), and [DemInf](baselines/demonstration-information/LICENSE).
