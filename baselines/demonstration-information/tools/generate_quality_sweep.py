import argparse
import copy
import itertools
import os

METHOD_TO_ESTIMATOR_MAP = {
    "infonce": ["nce"],
    "vip": ["vip"],
    "mine": ["mine"],
    "bc": ["l2", "stddev", "inv_stddev", "compatibility"],
    "vae": ["biksg", "ksg", "kl"],
}

ESTIMATOR_TO_MODEL_TYPE = {
    "ksg": ["obs", "action"],
    "biksg": ["obs", "action"],
    "kl": ["obs", "action"],
    "nce": ["obs_action"],
    "mine": ["obs_action"],
    "vip": ["obs"],
    "l2": ["obs"],
    "stddev": ["obs"],
    "inv_stddev": ["obs"],
    "compatibility": ["obs"],
}


METHOD_TO_CKPT_STEPS = {
    "state": {
        "infonce": {"obs_action": 20000},
        "vip": {"obs": 50000},
        "mine": {"obs_action": 50000},
        "bc": {"obs": 50000},
        "vae": {"obs": 50000, "action": 50000},
    },
    "image": {
        "infonce": {"obs_action": 40000},
        "vip": {"obs": 100000},
        "mine": {"obs_action": 60000},
        "bc": {"obs": 100000},
        "vae": {"obs": 100000, "action": 50000},
    },
}

BATCH_SIZE = 1024

ENV_KEY = "env"

# Can optionally specify a list of configs to group models over (e.g. beta)
ADDITIONAL_GROUPS = []
SPLIT_BY = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--grep", type=str, default=None)
    parser.add_argument("--mode", type=str, required=True)

    # Args for generating slurm scripts
    parser.add_argument("--save_split", type=int, default=None)
    parser.add_argument("--split_dir", type=str, default="run_scripts")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--no_tf", action="store_true", default=False)

    args = parser.parse_args()

    assert args.mode in ["state", "image"], "Must specify mode for checkpoint selection"
    METHOD_TO_CKPT_STEPS = METHOD_TO_CKPT_STEPS[args.mode]

    if not args.no_tf:
        import tensorflow as tf

        isdir = tf.io.gfile.isdir
        listdir = tf.io.gfile.listdir
        glob = tf.io.gfile.glob
    else:
        import glob as glob_module

        isdir = os.path.isdir
        listdir = os.listdir
        glob = glob_module.glob

    files = listdir(args.path)
    files = sorted(files)

    file_map = dict()
    run_set = set()
    for f in files:
        if not f.startswith("config-"):
            continue
        assert f.startswith("config-"), "Config type was not properly specified for: " + f
        method = f[len("config-") :].split("_")[0]
        assert method in METHOD_TO_ESTIMATOR_MAP, "Method could not be found."
        # Get the env name
        parts = f.split("-")
        idx = next(iter(i for i, p in enumerate(parts) if p.endswith(ENV_KEY)))
        env = "_".join(parts[idx + 1].split("_")[:-1])
        seed = int(f.split("seed-")[1][:1])
        if "type-" in f:
            model_type = {"s": "obs", "a": "action", "sa": "obs_action"}[f.split("type-")[1].split("_")[0]]
            assert model_type in ESTIMATOR_TO_MODEL_TYPE[METHOD_TO_ESTIMATOR_MAP[method][0]]
        else:
            assert len(ESTIMATOR_TO_MODEL_TYPE[METHOD_TO_ESTIMATOR_MAP[method][0]]) == 1
            model_type = ESTIMATOR_TO_MODEL_TYPE[METHOD_TO_ESTIMATOR_MAP[method][0]][0]

        if args.mode == "image" and "image" not in env:
            env = env.replace("mh", "mh_image")

        key = (
            env,
            method,
            seed,
        )
        additional_key = ()
        if len(ADDITIONAL_GROUPS) > 0:
            for group in ADDITIONAL_GROUPS:
                if group + "-" in f:
                    value = f.split(group + "-")[1].split("_")[0]
                    additional_key += (f"{group}-{value}",)
                # else:
                # print(f"Warning: group {group} not found in filename {f}")

        if key not in file_map:
            file_map[key] = dict()
        if additional_key not in file_map[key]:
            file_map[key][additional_key] = dict()
        if model_type not in file_map[key][additional_key]:
            file_map[key][additional_key][model_type] = []
        file_map[key][additional_key][model_type].append(f)
        run_set.add((env, method, seed))

    commands = []

    for env, method, seed in sorted(list(run_set)):
        for estimator in METHOD_TO_ESTIMATOR_MAP[method]:
            base_command_str = ["python scripts/quality/estimate_quality.py"]
            base_command_str.append("--estimator=" + estimator)
            # Add the model flag
            for additional_key in file_map[(env, method, seed)]:
                model_list = dict()
                for model_type in ESTIMATOR_TO_MODEL_TYPE[estimator]:
                    for _f in file_map[(env, method, seed)][additional_key][model_type]:
                        f = os.path.join(args.path, _f, str(METHOD_TO_CKPT_STEPS[method][model_type]))
                        if model_type not in model_list:
                            model_list[model_type] = []
                        model_list[model_type].append("--" + model_type + "_ckpt=" + f)

                model_list_keys = ESTIMATOR_TO_MODEL_TYPE[estimator]
                model_list_values = [model_list[k] for k in model_list_keys]
                model_args_list = list(itertools.product(*model_list_values))
                model_args_list = [
                    dict(zip(model_list_keys, combination, strict=False)) for combination in model_args_list
                ]

                for model_args in model_args_list:
                    command_str = copy.copy(base_command_str)
                    for model_type in model_list_keys:
                        command_str.append(model_args[model_type])
                    command_str.append("--batch_size=" + str(BATCH_SIZE))
                    # Carefully write the path. We will split on env and method
                    estimator_with_additional_groups = estimator
                    if len(additional_key) > 0:
                        estimator_with_additional_groups += "_" + "_".join(additional_key)
                    for split_by in SPLIT_BY:
                        for model_type in model_list_keys:
                            if split_by + "-" in model_args[model_type]:
                                estimator_with_additional_groups += (
                                    "_"
                                    + model_type
                                    + "_"
                                    + split_by
                                    + model_args[model_type].split(split_by + "-")[1].split("_")[0]
                                )
                    path = os.path.join(args.output, env, estimator_with_additional_groups, "seed-" + str(seed))
                    command_str.append("--path=" + path)

                    command = " ".join(command_str)
                    commands.append(command)

    if args.grep is not None:
        commands = [c for c in commands if args.grep in c]

    for command in commands:
        print(command)
        print()

    # You can then pipe the output of this command to a script, like > inference.sh

    # Alternatively, use the save_split flag to save the commands to multiple files
    # and optionally specify a path to a slurm prefix to add to the beginning of each file
    if args.save_split is not None:
        prefix = ""
        if args.prefix is not None:
            # Load slurm prefix from path
            with open(args.prefix, "r") as f:
                prefix = f.read()
        for split in range(args.save_split):
            output_file = os.path.join(f"{args.split_dir}/inference_{split}.sh")
            with open(output_file, "w") as f:
                f.write(prefix)
                for command in commands[
                    split * len(commands) // args.save_split : (split + 1) * len(commands) // args.save_split
                ]:
                    f.write(command + "\n")
