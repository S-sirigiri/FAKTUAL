import gymnasium as gym
import numpy as np
import requests

from openx.data.utils import StateEncoding

SERVER_URL = "" # URL to the server that will predict actions.


def convert_numpy_to_list(d):
    if isinstance(d, dict):
        return {key: convert_numpy_to_list(value) for key, value in d.items()}
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def convert_list_to_numpy(d):
    if isinstance(d, dict):
        return {key: convert_list_to_numpy(value) for key, value in d.items()}
    if isinstance(d, list):
        return np.array(d)
    return d


class ALOHAGymClient(gym.Env):
    def __init__(self, image_shape):
        super().__init__()

        self.observation_space = gym.spaces.Dict(
            dict(
                image=gym.spaces.Dict(
                    dict(
                        high=gym.spaces.Box(shape=image_shape, dtype=np.uint8, low=0, high=255),
                        left_wrist=gym.spaces.Box(shape=image_shape, dtype=np.uint8, low=0, high=255),
                        right_wrist=gym.spaces.Box(shape=image_shape, dtype=np.uint8, low=0, high=255),
                    )
                ),
                state=gym.spaces.Dict(
                    {
                        StateEncoding.JOINT_POS: gym.spaces.Box(
                            shape=(12,), low=-np.inf, high=np.inf, dtype=np.float32
                        ),
                        StateEncoding.GRIPPER: gym.spaces.Box(shape=(2,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
            )
        )

        self.action_space = gym.spaces.Dict(
            dict(
                desired_delta=gym.spaces.Dict(
                    {
                        StateEncoding.JOINT_POS: gym.spaces.Box(
                            shape=(12,), low=-np.inf, high=np.inf, dtype=np.float32
                        ),
                    }
                ),
                desired_absolute=gym.spaces.Dict(
                    {
                        StateEncoding.JOINT_POS: gym.spaces.Box(
                            shape=(12,), low=-np.inf, high=np.inf, dtype=np.float32
                        ),
                        StateEncoding.GRIPPER: gym.spaces.Box(shape=(2,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
            )
        )

    def wrap_env(self, structure, dataset_statistics, n_obs, n_action, exec_horizon, augment_kwargs):
        requests.post(
            f"{SERVER_URL}/wrap_env",
            json=dict(
                structure=structure,
                dataset_statistics=convert_numpy_to_list(dataset_statistics),
                n_obs=n_obs,
                n_action=n_action,
                exec_horizon=exec_horizon,
                augment_kwargs=augment_kwargs,
            ),
        )
        return

    def reset(self, *args, **kwargs):
        response = requests.post(f"{SERVER_URL}/reset")
        response = response.json()
        response["obs"] = convert_list_to_numpy(response["obs"])
        return response["obs"], response["info"]

    def step(self, action):
        action = convert_numpy_to_list(np.array(action))
        response = requests.post(f"{SERVER_URL}/step", json={"action": action})
        response = response.json()
        response["obs"] = convert_list_to_numpy(response["obs"])
        return response["obs"], response["reward"], response["terminated"], response["truncated"], response["info"]

    def save(self, path, filename):
        requests.post(f"{SERVER_URL}/save", json={"path": path, "filename": filename})
        return
