# Define the config for robomimic
import os

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.data.filters import filter_by_scores
from openx.data.utils import NormalizationType, StateEncoding
from openx.envs.robomimic import RobomimicEnv
from openx.networks.action_heads.continuous import L2ActionHead
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "square/mh,50,ksg,1"):
    # Parse the config string -- used for sweeping.
    # Define the structure
    env, percentile, estimator, seed = config_str.split(",")
    filter_path = os.path.join(
        "path/to/robomimic_image/inference/",
        env.replace("/", "_"),
        estimator,
        "seed-" + str(((int(seed) - 1) % 3) + 1),
        env.replace("/", "_") + ".pkl",
    )

    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.NONE,
                StateEncoding.EE_QUAT: NormalizationType.NONE,
                StateEncoding.GRIPPER: NormalizationType.NONE,
            },
            "image": {
                "agent": (84, 84),
                "wrist": (84, 84),
            },
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.BOUNDS,
                StateEncoding.EE_EULER: NormalizationType.BOUNDS,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.BOUNDS},
        },
    }

    dataloader = dict(
        datasets={
            env.replace("/", "_"): dict(
                path="path/to/datasets/robomimic_rlds_v2/{env}/1.0.0".format(env=env),
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(robomimic_dataset_transform),
                train_filter=ModuleSpec.create(filter_by_scores, filter_path, "ep_idx", percentile=float(percentile)),
            ),
        },
        n_obs=1,
        n_action=4,
        augment_kwargs=dict(scale_range=(0.85, 1.0), aspect_ratio_range=None),
        shuffle_size=100000,
        batch_size=128,
        recompute_statistics=False,
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BehaviorCloning,
        observation_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->image->wrist": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
            },
            trunk=ModuleSpec.create(
                Concatenate,
                model=ModuleSpec.create(MLP, [512, 512, 512], dropout_rate=0.5, activate_final=True),
                flatten_time=True,
            ),
        ),
        action_head=ModuleSpec.create(
            L2ActionHead,
            model=None,
            action_dim=7,
            action_horizon=4,
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=0.0001,
        warmup_steps=1000,
        decay_steps=400000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw)

    envs = {
        env.replace("/", "_"): ModuleSpec.create(
            RobomimicEnv,
            path="path/to/datasets/robomimic/{env}/image.hdf5".format(env=env),
            horizon=400,
        )
    }
    return ConfigDict(
        dict(
            structure=structure,
            envs=envs,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            # Add training parameters
            steps=100001,
            log_freq=1000,
            val_freq=10000,
            eval_freq=100000,
            save_freq=100000,
            val_steps=32,
            n_eval_proc=50,
            eval_ep=200,
            exec_horizon=2,
            seed=int(seed),
        )
    )
