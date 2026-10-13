"""
Config for training beta vae's on robomimic

Example command:

python scripts/train.py \
    --config=configs/quality/vae_droid.py:square/mh,s \
    --path test \
    --name state
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.oxe import droid_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18, ResNet18Decoder
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="pen_in_cup,sa,1"):
    ds, config_type, seed = config_str.split(",")
    seed = int(seed)
    assert config_type in {"s", "a", "sa"}
    encoder_keys = {
        "s": {
            "observation->state": None,
            "observation->image->agent_1": ModuleSpec.create(ResNet18, num_kp=64),
            "observation->image->wrist": ModuleSpec.create(ResNet18, num_kp=64),
        },
        "a": {"action": None},
        "sa": {
            "observation->state": None,
            "observation->image->agent_1": ModuleSpec.create(ResNet18, num_kp=64),
            "observation->image->wrist": ModuleSpec.create(ResNet18, num_kp=64),
            "action": None,
        },
    }[config_type]

    decoder_keys = {
        "s": {
            "observation->state": None,
            "observation->image->agent_1": ModuleSpec.create(ResNet18Decoder),
            "observation->image->wrist": ModuleSpec.create(ResNet18Decoder),
        },
        "a": {"action": None},
        "sa": {
            "observation->state": None,
            "observation->image->agent_1": ModuleSpec.create(ResNet18Decoder),
            "observation->image->wrist": ModuleSpec.create(ResNet18Decoder),
            "action": None,
        },
    }[config_type]
    seed = int(seed)

    z_dim = {
        "s": 24,
        "a": 16,  # XYZ+ROT=6
        "sa": 40,
    }[config_type]

    weights = {
        "s": {"observation->state": 1.0, "observation->image->agent_1": 1 / 200, "observation->image->wrist": 1 / 200},
        "a": {"action": 1.0},
        "sa": {
            "observation->state": 1.0,
            "observation->image->agent_1": 1 / 200,
            "observation->image->wrist": 1 / 200,
            "action": 1.0,
        },
    }[config_type]

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
        recompute_statistics=False,
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BetaVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders=encoder_keys,
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [1024, 1024], activate_final=True), flatten_time=True
            ),
        ),
        decoder=ModuleSpec.create(
            MultiDecoder,
            trunk=ModuleSpec.create(MLP, [1024, 1024], activate_final=True),
            decoders=decoder_keys,
        ),
        z_dim=z_dim,
        beta=0.01,
        weights=weights,
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
