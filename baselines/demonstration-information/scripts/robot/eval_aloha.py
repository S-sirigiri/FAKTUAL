import os
from datetime import datetime

import jax
import tensorflow as tf
from absl import app, flags
from jax.experimental import compilation_cache

from openx.envs.aloha_client import ALOHAGymClient
from openx.utils.evaluate import load_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "Path to checkpoint folder.")
flags.DEFINE_bool("show_image", True, "Whether or not to display the robot images.")
flags.DEFINE_string("dataset", None, "The name of the dataset to get the dataset statistics from")
flags.DEFINE_string("video_save_path", None, "whether or not to save videos.")
flags.DEFINE_string("task", None, "description of current task")
flags.DEFINE_integer("max_steps", 100, "Maximum number of steps to run the robot before terminating.")


def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Load
    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")
    rep_spec = jax.sharding.PartitionSpec()
    rep_sharding = jax.sharding.NamedSharding(mesh, rep_spec)
    alg, state, dataset_statistics, config = load_checkpoint(FLAGS.path, sharding=rep_sharding)
    if FLAGS.dataset is None:
        if len(dataset_statistics) > 1:
            raise ValueError(
                "Trained on multiple datasets, must provide key for statistics, choose one of: "
                + ", ".join(dataset_statistics.keys())
            )
        dataset_statistics = dataset_statistics[next(iter(dataset_statistics.keys()))]
    else:
        dataset_statistics = dataset_statistics[FLAGS.dataset]
    rng = jax.random.PRNGKey(config.seed)

    ### Define the Predict Function ###
    @jax.jit
    def predict(obs, rng):
        batch = dict(observation=obs)
        batch = jax.tree.map(lambda x: x[None], batch)
        action = alg.predict(state, batch, rng)
        return jax.tree.map(lambda x: x[0], action)

    ### Setup Eval Envs ###
    structure = config.structure.to_dict()
    n_obs, n_action = config.dataloader.n_obs, config.dataloader.n_action
    augment_kwargs = config.dataloader.to_dict().get("augment_kwargs", dict())

    env = ALOHAGymClient((240, 320))

    env.wrap_env(
        structure=structure,
        dataset_statistics=dataset_statistics,
        n_obs=n_obs,
        n_action=n_action,
        exec_horizon=n_action // 2,
        augment_kwargs=augment_kwargs,
    )

    obs, _ = env.reset()
    while input("Quit?\n") != "q":
        image = obs["image"]["high"]
        steps = 0
        done, trunc = False, False

        images = [image]
        try:
            while not done and not trunc:
                rng = jax.random.fold_in(rng, steps)
                action = predict(obs, rng)
                obs, _, done, trunc, _ = env.step(action)
                image = obs["image"]["high"]
                images.append(image)

                steps += 1
                trunc = trunc or steps == FLAGS.max_steps
        except KeyboardInterrupt:
            print("Ended early")

        if FLAGS.video_save_path is not None:
            checkpoint_name = (
                (FLAGS.path).split("/")[-2] if len((FLAGS.path).split("/")[-1]) < 2 else (FLAGS.path).split("/")[-1]
            )
            path = FLAGS.video_save_path + "/" + checkpoint_name + "/" + FLAGS.task
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
            env.save(path, filename)
            obs, _ = env.reset()


if __name__ == "__main__":
    app.run(main)
