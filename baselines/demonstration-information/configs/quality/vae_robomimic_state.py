"""
Config for training beta vae's on robomimic

Example command:

python scripts/train.py \
    --config=configs/quality/vae_robomimic_state.py:square/mh,sa \
    --path ../test/robomimic_vae/can \
    --name state
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="square/mh,sa,seed"):
    ds, config_type, seed = config_str.split(",")
    assert config_type in {"s", "a", "sa"}
    vae_keys = {
        "s": {"observation->state": None},
        "a": {"action": None},
        "sa": {"observation->state": None, "action": None},
    }[config_type]
    z_dim = {
        "s": 12,  # XYZ+ROT=6 + Object XYZ + Rot=6 + Gripper=1 -- Total 13
        "a": 6,  # XYZ+ROT=6+ Gripper -- Total 7
        "sa": 18,
    }[config_type]
    seed = int(seed)

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_QUAT: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.NONE,
                StateEncoding.MISC: NormalizationType.GAUSSIAN,
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
        recompute_statistics=False,  # Small, just recompute.
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BetaVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders=vae_keys,
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [512, 512], activate_final=True), flatten_time=True
            ),
        ),
        decoder=ModuleSpec.create(
            MultiDecoder,
            trunk=ModuleSpec.create(MLP, [512, 512], activate_final=True),
            decoders=vae_keys,
        ),
        z_dim=z_dim,
        beta=0.05,
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
            steps=50000,
            log_freq=500,
            val_freq=2500,
            save_freq=10000,
            val_steps=25,
            seed=seed,
        )
    )
