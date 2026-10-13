import os

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.robocrowd import (
    robocrowd_dataset_transform_image,
    robocrowd_dataset_transform_image_left,
    robocrowd_dataset_transform_image_right,
)
from openx.data.filters import filter_by_scores
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads.continuous import L1ActionHead
from openx.networks.components.aloha import ACTDecoder
from openx.networks.components.resnet import ResNet18
from openx.networks.components.vit import ViT
from openx.networks.core import MultiEncoder, Tokenize
from openx.utils.spec import ModuleSpec

# Use 3 images for policies.
IMAGE_KEYS_BY_ARMS = {
    "both": ["high", "left_wrist", "right_wrist"],
    "left": ["high", "left_wrist", "right_wrist"],
    "right": ["high", "left_wrist", "right_wrist"],
}


def get_config(config_str="hi_chew/mh_image,50,ksg,1"):
    if len(config_str.split(",")) == 4:
        # Append default values to config_str
        ds = config_str.split(",")[0]

        # Use full action space for policies.
        defaults = {
            "hi_chew/mh_image": "both,14",
            "hi_chew/mh_image_subtraj": "both,14",
            "tootsie_roll/mh_image": "both,14",
            "tootsie_roll/mh_image_subtraj": "both,14",
            "hershey_kiss/mh_image": "both,14",
            "hershey_kiss/mh_image_subtraj": "both,14",
        }
        config_str = ",".join([config_str, defaults[ds]])

    env, percentile, estimator, seed, arms, action_dim = config_str.split(",")
    filter_path = os.path.join(
        "path/to/output/scores",
        env.replace("/", "_"),
        estimator,
        "seed-" + seed,
        env.replace("/", "_") + ".pkl",
    )
    seed = int(seed)
    percentile = float(percentile)
    assert arms == "both"
    action_dim = int(action_dim)
    action_horizon = 100

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.GAUSSIAN,
            },
            # Remember: Image size is (height, width)
            "image": {image_key: (240, 320) for image_key in IMAGE_KEYS_BY_ARMS[arms]},
        },
        "action": {
            "desired_absolute": {
                StateEncoding.GRIPPER: NormalizationType.BOUNDS,
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,
            },
        },
    }

    dataloader = dict(
        datasets={
            ds.replace("/", "_"): dict(
                path="path/to/robocrowd/{ds}/1.0.0".format(ds=ds),
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(
                    robocrowd_dataset_transform_image
                    if arms == "both"
                    else robocrowd_dataset_transform_image_right
                    if arms == "right"
                    else robocrowd_dataset_transform_image_left
                ),
                train_filter=ModuleSpec.create(filter_by_scores, filter_path, "ep_idx", percentile=float(percentile)),
            )
        },
        n_obs=1,
        n_action=action_horizon,
        augment_kwargs=dict(scale_range=(0.9, 0.95), aspect_ratio_range=(1.33, 1.333)),  # Squish a little bit.
        shuffle_size=100000,
        batch_size=64,
        recompute_statistics=False,  # Small, just recompute.
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BehaviorCloning,
        observation_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->state": None,
            }
            | {
                "observation->image->" + image_key: ModuleSpec.create(ResNet18)
                for image_key in IMAGE_KEYS_BY_ARMS[arms]
            },
            trunk=ModuleSpec.create(
                Tokenize,
                embed_dim=512,
                flatten_time=True,
                model=ModuleSpec.create(
                    ViT,
                    num_layers=4,
                    num_heads=8,
                    mlp_dim=3200,
                    dropout_rate=0.1,
                    pool_type=None,
                ),
            ),
        ),
        action_head=ModuleSpec.create(
            L1ActionHead,
            model=ModuleSpec.create(
                ACTDecoder, action_horizon=action_horizon, num_layers=7, num_heads=8, mlp_dim=3200, dropout_rate=0.1
            ),
            action_dim=action_dim,
            action_horizon=None,
        ),
    )

    lr_schedule = ModuleSpec.create(optax.constant_schedule, 1e-5)
    # Weight decay taken from https://github.com/MarkFzp/act-plus-plus/blob/26bab0789d05b7496bacef04f5c6b2541a4403b5/detr/main.py#L17
    optimizer = ModuleSpec.create(optax.adamw, weight_decay=1e-4)

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=300000,
            log_freq=500,
            val_freq=5000,
            save_freq=20000,
            val_steps=25,
            seed=seed,
        )
    )
