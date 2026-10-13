from typing import Dict

import tensorflow as tf

from openx.data.utils import RobotType, StateEncoding


def robocrowd_dataset_transform(ep: Dict, arms: str, images: bool):
    action = ep["action"]
    abs_left_joint_action, abs_left_gripper_action, abs_right_joint_action, abs_right_gripper_action = (
        action[..., :6],
        action[..., 6:7],
        action[..., 7:13],
        action[..., 13:],
    )
    if images:
        if arms == "both":
            image_observation = {
                "image": {
                    "high": ep["observation"]["high_image"],
                    # "low": ep["observation"]["low_image"],
                    "left_wrist": ep["observation"]["left_wrist_image"],
                    "right_wrist": ep["observation"]["right_wrist_image"],
                },
            }
        elif arms == "left":
            image_observation = {
                "image": {
                    "high": ep["observation"]["high_image"],
                    "left_wrist": ep["observation"]["left_wrist_image"],
                },
            }
        elif arms == "right":
            image_observation = {
                "image": {
                    "high": ep["observation"]["high_image"],
                    "right_wrist": ep["observation"]["right_wrist_image"],
                },
            }
    else:
        image_observation = {}
    if arms == "both":
        state_observation = {
            "state": {
                StateEncoding.EE_POS: tf.concat(
                    (
                        ep["observation"]["state"]["left_ee_pos"],
                        ep["observation"]["state"]["right_ee_pos"],
                    ),
                    axis=-1,
                ),
                StateEncoding.JOINT_POS: tf.concat(
                    (
                        ep["observation"]["state"]["left_joint_pos"],
                        ep["observation"]["state"]["right_joint_pos"],
                    ),
                    axis=-1,
                ),
                StateEncoding.JOINT_VEL: tf.concat(
                    (
                        ep["observation"]["state"]["left_joint_vel"],
                        ep["observation"]["state"]["right_joint_vel"],
                    ),
                    axis=-1,
                ),
                StateEncoding.GRIPPER: tf.concat(
                    (
                        ep["observation"]["state"]["left_gripper_pos"],
                        ep["observation"]["state"]["right_gripper_pos"],
                    ),
                    axis=-1,
                ),
                StateEncoding.MISC: tf.concat(
                    (
                        ep["observation"]["state"]["left_object"],
                        ep["observation"]["state"]["right_object"],
                    ),
                    axis=-1,
                ),
            },
        }
    else:
        assert arms in ["left", "right"]
        state_observation = {
            "state": {
                StateEncoding.EE_POS: ep["observation"]["state"][f"{arms}_ee_pos"],
                StateEncoding.JOINT_POS: ep["observation"]["state"][f"{arms}_joint_pos"],
                StateEncoding.JOINT_VEL: ep["observation"]["state"][f"{arms}_joint_vel"],
                StateEncoding.GRIPPER: ep["observation"]["state"][f"{arms}_gripper_pos"],
                StateEncoding.MISC: ep["observation"]["state"][f"{arms}_object"],
            },
        }

    observation = image_observation | state_observation

    if arms == "both":
        action = {
            "desired_delta": {
                StateEncoding.JOINT_POS: tf.concat(
                    (
                        abs_left_joint_action - ep["observation"]["state"]["left_joint_pos"],
                        abs_right_joint_action - ep["observation"]["state"]["right_joint_pos"],
                    ),
                    axis=-1,
                ),
            },
            "desired_absolute": {
                StateEncoding.JOINT_POS: tf.concat(
                    (
                        abs_left_joint_action,
                        abs_right_joint_action,
                    ),
                    axis=-1,
                ),
                StateEncoding.GRIPPER: tf.concat((abs_left_gripper_action, abs_right_gripper_action), axis=-1),
            },
        }
    elif arms == "left":
        action = {
            "desired_delta": {
                StateEncoding.JOINT_POS: abs_left_joint_action - ep["observation"]["state"]["left_joint_pos"],
            },
            "desired_absolute": {
                StateEncoding.JOINT_POS: abs_left_joint_action,
                StateEncoding.GRIPPER: abs_left_gripper_action,
            },
        }
    elif arms == "right":
        action = {
            "desired_delta": {
                StateEncoding.JOINT_POS: abs_right_joint_action - ep["observation"]["state"]["right_joint_pos"],
            },
            "desired_absolute": {
                StateEncoding.JOINT_POS: abs_right_joint_action,
                StateEncoding.GRIPPER: abs_right_gripper_action,
            },
        }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.ALOHA
    ep["ep_idx"] = ep["episode_metadata"]["ep_idx"]

    ep["quality_score"] = ep["episode_metadata"]["quality_binary"]
    ep["quality_score_continuous"] = ep["episode_metadata"]["quality_continuous"]

    return ep


def robocrowd_dataset_transform_image(ep: Dict):
    return robocrowd_dataset_transform(ep, arms="both", images=True)


def robocrowd_dataset_transform_image_right(ep: Dict):
    return robocrowd_dataset_transform(ep, arms="right", images=True)


def robocrowd_dataset_transform_image_left(ep: Dict):
    return robocrowd_dataset_transform(ep, arms="left", images=True)


def robocrowd_dataset_transform_state(ep: Dict):
    return robocrowd_dataset_transform(ep, arms="both", images=False)


def robocrowd_dataset_transform_state_right(ep: Dict):
    return robocrowd_dataset_transform(ep, arms="right", images=False)


def robocrowd_dataset_transform_state_left(ep: Dict):
    return robocrowd_dataset_transform(ep, arms="left", images=False)
