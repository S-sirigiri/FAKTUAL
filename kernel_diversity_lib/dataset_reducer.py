import argparse
import ast
import sys
import json
from pathlib import Path

from rich import print as rprint

import numpy as np

if __package__:
    from .maximizer import (BlackBoxMaximizer, SubmodularMaximizer,
                            NonMonotoneSubmodularMaximizer,
                            DeterminantalPointProcessesSampling, MixedObjectiveMaximizer)
    from .diversity_metrics import (ShannonEntropy, VonNeumannEntropy,
                                    VendiScore, LogDeterminant, Determinant)
    from .kernels import KernelMatrix, ConvexKernelMatrix
    from .utils import kern_utils
    from .HDF5_handler import HDF5DatasetReducer
    from .hflerobot_dataset_handler import HFLeRobotReducer
else:
    from maximizer import (BlackBoxMaximizer, SubmodularMaximizer,
                           NonMonotoneSubmodularMaximizer,
                           DeterminantalPointProcessesSampling, MixedObjectiveMaximizer)
    from diversity_metrics import (ShannonEntropy, VonNeumannEntropy,
                                   VendiScore, LogDeterminant, Determinant)
    from kernels import KernelMatrix, ConvexKernelMatrix
    from utils import kern_utils
    from HDF5_handler import HDF5DatasetReducer
    from hflerobot_dataset_handler import HFLeRobotReducer


def _json_object(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {error.msg}") from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object keyed by modality name.")
    return result


def _key_list(values):
    """Accept space-separated keys or a quoted bracketed, comma-separated list."""
    text = ' '.join(values).strip()
    if text.startswith('[') or text.endswith(']'):
        if not (text.startswith('[') and text.endswith(']')):
            raise ValueError('Bracketed key lists must start with [ and end with ]')
        try:
            keys = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Also accept the convenient unquoted form: [key1, key2].
            contents = text[1:-1].strip()
            keys = [key.strip() for key in contents.split(',')] if contents else []
            if any(any(char in key for char in "'\"[]") or len(key.split()) != 1
                   for key in keys):
                raise ValueError('Use space-separated keys or a list like [key1, key2]')
    else:
        keys = values
    if not isinstance(keys, list) or any(not isinstance(key, str) or not key.strip()
                                         for key in keys):
        raise ValueError('Keys must be nonempty strings')
    if len(set(keys)) != len(keys):
        raise ValueError('Keys must be unique')
    return keys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Dataset reducer CLI')
    parser.add_argument('--dataset-type', default='robomimic',
                        choices=['robomimic', 'lerobot'],
                        help='Type of dataset to be reduced')
    parser.add_argument('--input', '-i', required=True, help='Path to input HDF5 file or LeRobot/Hugging Face repo ID')
    parser.add_argument('--output-suffix', '-o', default='reduced', help='Suffix for output filename')
    parser.add_argument('--output-path', default=None,
                        help='Output directory for HDF5, or output JSON path for LeRobot')
    parser.add_argument('--get-score', action='store_true', help='Only compute diversity score and exit')
    parser.add_argument('--get-score-after-maximizing', action='store_true', help='Compute diversity score after maximizing')
    parser.add_argument('--keys', nargs='+', default=None, metavar='KEY',
                        help='RoboMimic observation keys, space-separated or as "[key1, key2]"; '
                             'defaults to the handler observation keys')
    parser.add_argument('--numeric-keys', nargs='+', default=None, metavar='KEY',
                        help='LeRobot numeric keys, space-separated or as "[key1, key2]"; '
                             'use "[]" to disable numeric fields')
    parser.add_argument('--image-keys', nargs='+', default=None, metavar='KEY',
                        help='LeRobot image keys, space-separated or as "[key1, key2]"; '
                             'use "[]" to disable image embedding')
    parser.add_argument('--time-stride', type=int, default=1,
                        help='Sample every N time steps (positive integer; default: 1)')
    parser.add_argument('--include-last', action=argparse.BooleanOptionalAction, default=True,
                        help='Also keep the final frame when it is off the sampling grid')
    parser.add_argument('--episode-list', nargs='+', type=int, metavar='N',
                   help='List of episodes from the dataset that will be used e.g. --episode-list 1 2 3 4')

    # Kernel options
    parser.add_argument('--combination-method', '--combination',
                        choices=['concatenate', 'non-convex', 'convex'], default='concatenate',
                        help='Concatenate modality features (default), or combine modality kernels')
    parser.add_argument('--modality-weights', type=_json_object, default=None,
                        help='Convex weights as a JSON object; omitted modalities get zero weight')
    parser.add_argument('--modality-kernel-configs', type=_json_object, default=None,
                        help='JSON object of per-modality KernelMatrix parameter overrides')
    parser.add_argument('--secondary-modality-weights', type=_json_object, default=None,
                        help='Convex weights for the secondary mixed objective (defaults to primary weights)')
    parser.add_argument('--secondary-modality-kernel-configs', type=_json_object, default=None,
                        help='Per-modality overrides for the secondary mixed objective kernel')
    parser.add_argument('--renormalize-weights', action='store_true',
                        help='Normalize positive convex weights that do not sum to one')
    parser.add_argument('--kernel-type', default='Random fourier signature features kernel',
                        choices=[
                            'Signature kernel torch', 'Signature kernel cupy', 'Signature kernel',
                            'Random fourier signature features kernel',
                            'Random warping series kernel', 'Global alignment kernel'
                        ], help='Type of sequence kernel')
    parser.add_argument('--secondary-kernel-type', default='Random fourier signature features kernel',
                        choices=[
                            'Signature kernel torch', 'Signature kernel cupy', 'Signature kernel',
                            'Random fourier signature features kernel',
                            'Random warping series kernel', 'Global alignment kernel'
                        ], help='Type of sequence kernel')
    parser.add_argument('--bandwidth', type=float, default=None, help='Bandwidth for kernel')
    parser.add_argument('--n-levels', type=int, default=5, help='n_levels for Random Fourier Signature Features')
    parser.add_argument('--n-components', type=int, default=100, help='n_components for RFSF or RWS')
    parser.add_argument('--method', default='TRP', choices=['TRP', 'DP', 'TS'], help='Projection method for RFSF')
    parser.add_argument('--max-warp', type=int, default=None, help='max_warp for Random Warping Series')
    parser.add_argument('--stdev', type=float, default=None, help='stdev for Random Warping Series')
    parser.add_argument('--n-features', type=int, default=None, help='n_features for RWS or Global Alignment')
    parser.add_argument('--random-state', type=int, default=None, help='random_state for RWS or random baseline')
    parser.add_argument('--use-gpu', action='store_true', help='Flag to use GPU in RWS')
    parser.add_argument('--max-batch', type=int, default=None, help='max_batch for Signature kernel torch')
    parser.add_argument('--device', default=None, help='Device for Signature kernel torch')
    parser.add_argument('--dyadic-order', type=int, default=1, help='dyadic_order for Signature kernel torch')

    # Diversity metric
    parser.add_argument('--metric', choices=['shannon', 'von_neumann', 'vendi', 'logdet', 'det'],
                        help='Diversity metric to use')
    parser.add_argument('--secondary-metric', choices=['shannon', 'von_neumann', 'vendi', 'logdet', 'det'],
                        default='shannon', help='Secondary metric to use for mixed objective maximization')
    parser.add_argument('--vendi-method', choices=['shannon', 'von_neumann'], default='shannon',
                        help='Entropy type for VendiScore')

    # Maximizer options
    parser.add_argument('--maximizer', choices=['submodular', 'nonmonotone', 'blackbox', 'mixed-objective', 'DPP' , 'random', 'arrange'],
                        help='Selection strategy')
    parser.add_argument('--split-percentage', type=int, default=30, help='Split percentage for mixed objective maximization')
    parser.add_argument('--stochastic-greedy', action='store_true', help='Use stochastic greedy for submodular maximizer')
    parser.add_argument('--cross-entropy', action='store_true', help='Use cross-entropy for blackbox maximizer')
    parser.add_argument('--k', type=int, help='Cardinality constraint (number of demos to select)')
    parser.add_argument('--epsilon', type=float, default=0.01, help='epsilon for stochastic greedy')
    parser.add_argument('--num-samples', type=int, default=1000, help='num_samples for cross-entropy maximizer')
    parser.add_argument('--elite-frac', type=float, default=0.1, help='elite_frac for cross-entropy maximizer')
    parser.add_argument('--max-iter', type=int, default=50, help='max_iter for blackbox maximizers')
    parser.add_argument('--smoothing', type=float, default=0.7, help='smoothing for cross-entropy maximizer')

    # Embedding options
    parser.add_argument('--embedding', choices=['clip', 'dinov2'], default='clip', help='Embedding choice for the images')
    parser.add_argument('--embedding-dim', type=int, default=None, help='Embedding dimension for the images')

    args = parser.parse_args(argv)
    for option in ('keys', 'numeric_keys', 'image_keys'):
        values = getattr(args, option)
        if values is not None:
            try:
                setattr(args, option, _key_list(values))
            except ValueError as error:
                parser.error(f'--{option.replace("_", "-")}: {error}')
    if args.dataset_type == 'robomimic':
        if args.numeric_keys is not None or args.image_keys is not None:
            parser.error('--numeric-keys and --image-keys require --dataset-type lerobot')
        if args.keys == []:
            parser.error('--keys requires at least one observation key')
    else:
        if args.keys is not None:
            parser.error('--keys requires --dataset-type robomimic')
        if args.numeric_keys == [] and args.image_keys == []:
            parser.error('At least one numeric or image key is required')
        if args.numeric_keys is not None and args.image_keys is not None:
            if set(args.numeric_keys) & set(args.image_keys):
                parser.error('Numeric and image modality keys must be unique')
    convex_options = (args.modality_weights, args.modality_kernel_configs,
                      args.secondary_modality_weights, args.secondary_modality_kernel_configs)
    if args.combination_method != 'convex' and (any(v is not None for v in convex_options)
                                               or args.renormalize_weights):
        parser.error('Modality weights/configs require --combination-method convex')
    if args.time_stride <= 0:
        parser.error('--time-stride must be positive')
    if args.get_score_after_maximizing and args.metric is None:
        parser.error('--get-score-after-maximizing requires --metric')
    # If only scoring, no further args required
    if args.get_score:
        if args.metric is None:
            parser.error('Scoring requires --metric')
        return args

    # For reduction runs, ensure required args
    if args.maximizer is None:
        parser.error('Reduction requires --maximizer')
    if args.k is None:
        parser.error('Reduction requires --k')
    if args.maximizer not in ['random', 'DPP'] and args.metric is None:
        parser.error('Reduction requires --metric unless --maximizer random')
    if args.stochastic_greedy and args.maximizer != 'submodular':
        parser.error('--stochastic-greedy requires --maximizer submodular')
    if args.cross_entropy and args.maximizer != 'blackbox':
        parser.error('--cross-entropy requires --maximizer blackbox')
    return args


def build_kernel(args, data, reducer, *, secondary=False):
    """Build and fit either combination with identical global kernel options."""
    options = dict(
        kernel_type=args.secondary_kernel_type if secondary else args.kernel_type,
        max_batch=args.max_batch, device=args.device, dyadic_order=args.dyadic_order,
        bandwidth=args.bandwidth, n_levels=args.n_levels, n_components=args.n_components,
        method=args.method, max_warp=args.max_warp, stdev=args.stdev,
        n_features=args.n_features, random_state=args.random_state, use_gpu=args.use_gpu,
    )
    if args.combination_method == 'convex':
        weights = args.secondary_modality_weights if secondary else args.modality_weights
        if secondary and weights is None:
            weights = args.modality_weights
        configs = args.secondary_modality_kernel_configs if secondary else args.modality_kernel_configs
        km = ConvexKernelMatrix(
            reducer.modality_slices, weights, kernel_configs=configs,
            renormalize=args.renormalize_weights, **options,
        )
        km.fit(data)
    else:
        options['bandwidth'] = args.bandwidth or kern_utils.KernelUtilities.compute_bandwidth(data)
        km = KernelMatrix(**options)
        if hasattr(km.kernel, 'fit'):
            km.kernel.fit(data)
    return km


def build_metric(name, km, vendi_method='shannon'):
    """Use the same metric constructors for both primary and secondary kernels."""
    if name == 'vendi':
        return VendiScore(km, method=vendi_method)
    constructors = {'shannon': ShannonEntropy, 'von_neumann': VonNeumannEntropy,
                    'logdet': LogDeterminant, 'det': Determinant}
    return constructors[name](km)


def main():
    args = parse_args()
    if args.dataset_type == 'lerobot' and not args.get_score:
        out_path = Path(args.output_path or args.output_suffix).with_suffix(".json")
        if out_path.exists(): #and args.dataset_type == 'lerobot':  # or: if out_path.is_file():
            print(f"{out_path} already exists")
            sys.exit(0)
    rprint("[red]Loading the dataset\n[/red]")
    if args.dataset_type == 'robomimic':
        embedding = args.embedding
        if (args.maximizer == 'random' and args.keys is None
                and not (args.get_score or args.get_score_after_maximizing)):
            embedding = None
        reducer = HDF5DatasetReducer(args.input,
                                     embedding=embedding,
                                     embedding_dim=args.embedding_dim,
                                     time_stride=args.time_stride,
                                     keys=args.keys,
                                     include_last=args.include_last)
    elif args.dataset_type == 'lerobot':
        reducer = HFLeRobotReducer(repo_id=args.input,
                                   numeric_keys=args.numeric_keys,
                                   image_keys=args.image_keys,
                                   embedding=args.embedding,
                                   embedding_output_dim=args.embedding_dim,
                                   time_stride=args.time_stride,
                                   episodes_filter=args.episode_list,
                                   include_last=args.include_last)
    data = reducer.get_demos()

    # Score-only mode
    if args.get_score:
        km = build_kernel(args, data, reducer)
        metric = build_metric(args.metric, km, args.vendi_method)
        rprint(f"The dataset '{args.input}' has {data.shape[0]} train demos.")
        rprint(f"Score of dataset: {reducer.get_diversity_score(metric):.16f}")
        sys.exit(0)

    # Instantiate kernel + metric for reduction (if needed)
    km, metric = None, None
    if args.maximizer not in ('random', 'arrange') or args.get_score_after_maximizing:
        km = build_kernel(args, data, reducer)
        if args.metric is not None:
            metric = build_metric(args.metric, km, args.vendi_method)

    N = data.shape[0]
    if not 0 <= args.k <= N:
        raise ValueError(f"--k must be between 0 and the number of demos ({N}).")
    # Selection
    rprint(f"[red]\nSelecting diversity maximized subset of size [/red] {args.k}\n")
    if args.maximizer == 'submodular':
        maximizer = SubmodularMaximizer(metric, data)
        if args.stochastic_greedy:
            top_idxes = maximizer.stochastic_greedy(args.k, epsilon=args.epsilon)
        else:
            top_idxes = maximizer.lazy_greedy(args.k)
    elif args.maximizer == 'nonmonotone':
        maximizer = NonMonotoneSubmodularMaximizer(metric, data)
        top_idxes = maximizer.random_greedy(args.k)
    elif args.maximizer == 'blackbox':
        maximizer = BlackBoxMaximizer(metric, data)
        if args.cross_entropy:
            top_idxes = maximizer.cross_entropy_maximizer(
                args.k,
                num_samples=args.num_samples,
                elite_frac=args.elite_frac,
                max_iter=args.max_iter,
                smoothing=args.smoothing
            )
        else:
            top_idxes = maximizer.greedy_local_search(k=args.k, max_iters=args.max_iter)
    elif args.maximizer == 'DPP':
        maximizer = DeterminantalPointProcessesSampling(km, data)
        top_idxes = maximizer.sample(args.k)
    elif args.maximizer == 'mixed-objective':
        secondary_km = build_kernel(args, data, reducer, secondary=True)
        metric_two = build_metric(args.secondary_metric, secondary_km, args.vendi_method)
        maximizer = MixedObjectiveMaximizer(metric_one=metric, metric_two=metric_two, data=data)
        top_idxes = maximizer.maximize(k=args.k, split_precentage=args.split_percentage)
    elif args.maximizer == 'random':
        seed = args.random_state if args.random_state is not None else 1
        np.random.seed(seed)
        top_idxes = np.random.choice(N, size=args.k, replace=False).tolist()
    elif args.maximizer == 'arrange':
        top_idxes = np.arange(args.k).tolist()
    else:
        raise ValueError(f"Unknown maximizer: {args.maximizer}")

    if args.dataset_type == 'robomimic':
        # Reduce and write
        rprint("[red]Saving the new dataset[/red] \n")
        if args.get_score_after_maximizing:
            reducer.process(top_idxes, args.output_path, args.output_suffix, metric=metric)
        else:
            reducer.process(top_idxes, args.output_path, args.output_suffix, metric=None)
    elif args.dataset_type == 'lerobot':
        indx = list(reducer.map_back(list(top_idxes)))
        rprint(f"List of most {args.k} diverse episodes: \n{indx}")
        if args.get_score_after_maximizing and metric is not None:
            rprint(f"Score of original dataset: {metric(data):.16f}")
            rprint(f"Score of reduced dataset: {metric(data[list(top_idxes)]):.16f}")
        out_path = Path(args.output_path or args.output_suffix).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(indx, f, separators=(",", ":"), ensure_ascii=False)


if __name__ == '__main__':
    main()
