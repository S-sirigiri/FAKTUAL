"""
Config for training beta vae's on robomimic

Example command:

python scripts/train.py --config=configs/quality/vip_robomimic_state.py --path ../test/robomimic/ --name infonce

"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.mine import MINE
from openx.data.datasets.oxe import droid_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="pen_in_cup,1"):
    ds, seed = config_str.split(",")
    seed = int(seed)

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.GAUSSIAN,
            },
            "image": {
                "agent_1": (128, 128),
                "wrist": (128, 128),
            },
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.BOUNDS},
        },
    }

    dataloader = dict(
        datasets={
            ds.replace("/", "_"): dict(
                path="path/to/droid/{ds}/1.0.0".format(ds=ds),
                train_split="train",
                transform=ModuleSpec.create(droid_dataset_transform),
            ),
        },
        n_obs=1,
        n_action=4,
        augment_kwargs=dict(scale_range=(0.9, 0.95), aspect_ratio_range=(1.33, 1.333)),  # Squish a little bit.
        shuffle_size=100000,
        batch_size=256,
        n_step=1,  # Add next obs
        add_initial_observation=True,  # Add initial obs
        goal_conditioning="uniform",
        recompute_statistics=False,  # Small, just recompute.
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        MINE,
        obs_action_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->state": None,
                "observation->image->agent_1": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->image->wrist": ModuleSpec.create(ResNet18, num_kp=64),
                "action": None,
            },
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [1024, 1024], activate_final=True), flatten_time=True
            ),
        ),
        alpha=0.9,
    )

    lr_schedule = ModuleSpec.create(optax.constant_schedule, 0.0001)
    optimizer = ModuleSpec.create(optax.adam)

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=100000,
            log_freq=500,
            val_freq=2500,
            save_freq=10000,
            val_steps=25,
            seed=seed,
        )
    )
