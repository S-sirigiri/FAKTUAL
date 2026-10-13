"""
Config for training beta vae's on robomimic

Example command:

python scripts/train.py --config=configs/quality/infonce_robomimic_state.py --path ../test/robomimic/ --name infonce

"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads.continuous import L2ActionHead
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="square/mh,1"):
    ds, seed = config_str.split(",")
    seed = int(seed)

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_QUAT: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.GAUSSIAN,
            },
            "image": {
                "agent": (84, 84),
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
                path="path/to/robomimic_rlds_v2/{ds}/1.0.0".format(ds=ds),
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(robomimic_dataset_transform),
            ),
        },
        n_obs=1,
        n_action=1,
        shuffle_size=100000,
        batch_size=256,
        augment_kwargs=dict(scale_range=(0.85, 1.0), aspect_ratio_range=None),
        recompute_statistics=False,  # Small, just recompute.
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BehaviorCloning,
        observation_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={"observation->state": None, "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64)},
            trunk=ModuleSpec.create(
                Concatenate,
                model=None,  # Move model to later so we have a bigger ensemble.
                flatten_time=True,
            ),
        ),
        action_head=ModuleSpec.create(
            L2ActionHead,
            model=ModuleSpec.create(MLP, [512, 512], dropout_rate=0.5, activate_final=True),
            action_dim=7,
            action_horizon=1,
            ensemble_size=5,
        ),
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
