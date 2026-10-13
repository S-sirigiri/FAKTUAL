import collections
import math
import os
import pickle

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "The path to the scores pickle file, or a folder", required=True)
flags.DEFINE_string("title", None, "The title of the plot", required=False)
flags.DEFINE_string("output", "plot.png", "The output path of the plot", required=False)
flags.DEFINE_boolean("add_avg", False, "Whether or not to add an average plot.")
flags.DEFINE_boolean("use_tf", True, "Whether or not to import tensorflow for gfile.")
flags.DEFINE_boolean("use_continuous_quality", False, "Whether or not to use the continuous quality labels.")
flags.DEFINE_string("estimator", None, "Filter esimator", required=False)
flags.DEFINE_string("type", None, "Whether or not to use final formatting")
flags.DEFINE_string("order", None, "A dataset ordering. See DATASET ORDERS")
flags.DEFINE_boolean("legend", True, "Whether or not to have a legend", required=False)


class Color:
    Orange = np.array((255, 155, 0)) / 255
    Teal = np.array((0, 187, 177)) / 255
    DarkOrange = 0.8 * Orange
    LightOrange = np.array((255, 215, 150)) / 255
    LightGrey = np.array((185, 185, 185)) / 255
    Grey = np.array((127, 127, 127)) / 255
    DarkGrey = np.array((89, 89, 89)) / 255
    Black = np.array((0, 0, 0)) / 255

    Yellow = np.array([255, 225, 53]) / 255
    Green = np.array([55, 126, 77]) / 255
    LightGreen = np.array([220, 237, 218]) / 255

    Purple = np.array([138, 80, 182]) / 255
    Slate = np.array([106, 109, 124]) / 255
    Violet = np.array([127, 114, 143]) / 255
    Maroon = np.array([116, 38, 63]) / 255
    Brown = np.array([76, 39, 9]) / 255
    Blue = np.array([102, 189, 249]) / 255
    DarkBlue = np.array([16, 53, 81]) / 255
    MediumBlue = 1.8 * DarkBlue

    Red = np.array([242, 97, 88]) / 255
    LightBrown = np.array([194, 153, 121]) / 255
    Aqua = np.array([131, 232, 186]) / 255
    LightViolet = Violet * 1.3


ESTIMATOR_NAMES = {
    "ksg": "DemInf (Ours)",
    "vip": "VIP",
    "compatibility": "Compatibility",
    "stddev": "Uncertainty",
    "nce": "InfoNCE (MI)",
    "mine": "MINE (MI)",
    "ksg_5_8": "(5,6,7)",
    "ksg_2_5": "(2,3,4)",
    "ksg_8_11": "(8,9,10)",
    "ksg_z-1x": "s=16,a=6",
    "ksg_z-0.5x": "s=8,a=3",
    "ksg_z-1.5x": "s=24,a=9",
    "ksg_beta-0.001": "Beta 0.001",
    "ksg_beta-0.01": "Beta 0.01",
    "ksg_beta-0.1": "Beta 0.1",
    "biksg": "BiKSG",
    "kl": "KL",
    "l2": "Policy Loss",
    "absolute": "Absolute Actions",
    "delta": "Delta Actions",
}

COLORS = {
    "ksg": Color.Orange,
    "ksg_5_8": Color.Orange,
    "ksg_2_5": Color.LightOrange,
    "ksg_8_11": Color.DarkOrange,
    "ksg_z-1x": Color.Orange,
    "ksg_z-0.5x": Color.LightOrange,
    "ksg_z-1.5x": Color.DarkOrange,
    "ksg_beta-0.01": Color.Orange,
    "ksg_beta-0.001": Color.LightOrange,
    "ksg_beta-0.1": Color.DarkOrange,
    "biksg": Color.Red,
    "kl": Color.Maroon,
    "vip": Color.Purple,
    "compatibility": Color.Blue,  # Blue
    "stddev": Color.MediumBlue,
    "nce": Color.LightBrown,  # Brown
    "mine": Color.Red,  # Red
    "oracle": Color.Grey,  # Gray
    "random": Color.Teal,
    "l2": Color.Slate,
}

PLOT_TYPES = {
    "baseline": ["ksg", "vip", "compatibility", "stddev"],
    "mi": ["ksg", "mine", "nce"],
    "knn": ["ksg", "biksg", "kl"],
    "all": ["ksg", "kl", "mine", "nce", "vip", "compatibility", "stddev", "l2"],
}

ORDER = [
    "ksg",
    "ksg_5_8",
    "ksg_2_5",
    "ksg_8_11",
    "ksg_z-1x",
    "ksg_z-0.5x",
    "ksg_z-1.5x",
    "ksg_beta-0.01",
    "ksg_beta-0.1",
    "ksg_beta-0.001",
    "kl",
    "biksg",
    "mine",
    "nce",
    "vip",
    "compatibility",
    "stddev",
    "l2",
    "absolute",
    "delta",
]

CROP_POINT = 0.1  # Leave at least 10% of the data.

DATASET_ORDERS = {
    "robomimic": ["lift_mh", "can_mh", "square_mh"],
    "robomimic_image": ["lift_mh_image", "can_mh_image", "square_mh_image"],
    "robocrowd_image": [
        "hi_chew_mh_image_subtraj",
        "tootsie_roll_mh_image_subtraj",
        "hershey_kiss_mh_image_subtraj",
        "hi_chew_mh_image",
        "tootsie_roll_mh_image",
        "hershey_kiss_mh_image",
    ],
    "droid": ["rack", "pen_in_cup"],
}

DATASET_NAMES = {
    "lift_mh": "Lift MH",
    "can_mh": "Can MH",
    "square_mh": "Square MH",
    "lift_mh_image": "Lift MH Image",
    "can_mh_image": "Can MH Image",
    "square_mh_image": "Square MH Image",
    "hi_chew_mh": "HiChew Play",
    "tootsie_roll_mh": "TootsieRoll Play",
    "hershey_kiss_mh": "HersheyKiss Play",
    "hi_chew_mh_subtraj": "HiChew",
    "tootsie_roll_mh_subtraj": "TootsieRoll",
    "hershey_kiss_mh_subtraj": "HersheyKiss",
    "hi_chew_mh_image": "HiChew Play",
    "tootsie_roll_mh_image": "TootsieRoll Play",
    "hershey_kiss_mh_image": "HersheyKiss Play",
    "hi_chew_mh_image_subtraj": "HiChew",
    "tootsie_roll_mh_image_subtraj": "TootsieRoll",
    "hershey_kiss_mh_image_subtraj": "HersheyKiss",
    "pen_in_cup": "PenInCup",
    "rack": "DishRack",
}



def load_scores(path, estimator):
    with open(path, "rb") as f:
        scores = pickle.load(f)
        idxs = list(scores["ep_idx"].keys())
        # Use continuous scores if we have them
        if "quality_continuous_by_ep_idx" in scores:
            x = [scores["quality_continuous_by_ep_idx"][idx] for idx in idxs]
        else:
            x = [scores["quality_by_ep_idx"][idx] for idx in idxs]
        y = [scores["ep_idx"][idx] for idx in idxs]
    mult = -1 if estimator == "l2" else 1
    return np.array(x), mult * np.array(y)


def convert_to_threshold_curve(x, y):
    sort_idx = np.argsort(y)  # Sorts from low score to high score episodes.
    rev_sorted_quality_labels = x[sort_idx][::-1]  # The sorted quality labels
    total_quality_labels = np.cumsum(rev_sorted_quality_labels)
    num_data_points = 1 + np.arange(total_quality_labels.shape[0])
    avg_quality_label = total_quality_labels / num_data_points
    avg_quality_label = avg_quality_label[::-1]
    data_pts_removed = np.arange(avg_quality_label.shape[0])

    # Crop to the CROP_POINT
    crop = int((1 - CROP_POINT) * data_pts_removed.shape[0])
    data_pts_removed = data_pts_removed[:crop]
    avg_quality_label = avg_quality_label[:crop]

    return data_pts_removed, avg_quality_label


def plot_curve(ax, xs, ys, label=None, **kwargs):
    plot_df = pd.DataFrame({"x": np.concatenate(xs), "y": np.concatenate(ys)})
    sns.lineplot(
        ax=ax, x="x", y="y", data=plot_df, sort=True, errorbar="se" if len(xs) > 1 else None, label=label, **kwargs
    )


def main(_):
    if FLAGS.use_tf:
        import tensorflow as tf

        isdir = tf.io.gfile.isdir
        listdir = tf.io.gfile.listdir
        glob = tf.io.gfile.glob
    else:
        import glob as glob_module

        isdir = os.path.isdir
        listdir = os.listdir
        glob = glob_module.glob

    path = FLAGS.path

    sns.set_context(context="paper", font_scale=0.68)
    sns.set_style("white", {"font.family": "serif"})
    x_key, y_key = "Num Ep. Filtered", "Avg Quality Label"
    scale_factor = 1.5

    if isdir(path):
        # NOTE: folder is assumed to have
        # env/estimator/seed-{i}/scores_{env}.pkl
        # If you do not have this format, the script will fail.

        datasets = listdir(path)
        datasets = (
            sorted(datasets, key=lambda x: DATASET_ORDERS[FLAGS.order].index(x))
            if FLAGS.order is not None
            else sorted(datasets)
        )
        if FLAGS.add_avg:
            datasets = [*datasets, "average"]

        figsize = (scale_factor * len(datasets), scale_factor * (1.12 if FLAGS.legend else 1.06))
        fig, axes = plt.subplots(
            1, len(datasets), figsize=figsize, sharey=FLAGS.order is not None and FLAGS.order != "robocrowd_image"
        )
        all_xs, all_ys = collections.defaultdict(list), collections.defaultdict(list)

        for i, dataset in enumerate(datasets):
            ax = axes.flat[i] if len(datasets) > 1 else axes

            if dataset == "average":
                # Make the average plot, this is called last.
                for estimator in all_xs:
                    plot_curve(ax, all_xs[estimator], all_ys[estimator], label=estimator)
                x = all_xs[estimator]
            else:
                # Make the normal plot
                estimators = listdir(os.path.join(path, dataset))
                if FLAGS.estimator is not None and FLAGS.estimator != "":
                    estimators = [estimator for estimator in estimators if estimator.startswith(FLAGS.estimator)]
                # TODO: Override
                if FLAGS.type is not None:
                    estimators = [estimator for estimator in estimators if estimator in PLOT_TYPES[FLAGS.type]]
                estimators = sorted(estimators, key=lambda x: ORDER.index(x))
                for estimator in estimators:
                    seeds = glob(os.path.join(path, dataset, estimator, "*/*.pkl"))
                    xs, ys = list(
                        zip(
                            *(convert_to_threshold_curve(*load_scores(seed, estimator)) for seed in seeds), strict=False
                        )
                    )
                    plot_curve(
                        ax,
                        xs,
                        ys,
                        label=ESTIMATOR_NAMES.get(estimator, estimator),
                        color=COLORS.get(estimator),
                    )
                    all_xs[estimator].extend(xs)
                    all_ys[estimator].extend(ys)

            # Add the oracle
            quality_labels, _ = load_scores(seeds[0], estimator)
            x, y = convert_to_threshold_curve(quality_labels, quality_labels)
            sns.lineplot(
                ax=ax,
                x="x",
                y="y",
                data=pd.DataFrame({"x": x, "y": y}),
                sort=True,
                errorbar=None,
                label="Oracle",
                color=COLORS["oracle"],
                linestyle="dashed",
                alpha=0.6,
            )
            # Add the random baseline
            sns.lineplot(
                ax=ax,
                x="x",
                y="y",
                data=pd.DataFrame({"x": x, "y": np.mean(quality_labels)}),
                errorbar=None,
                label="Random",
                color=COLORS["random"],
                alpha=0.6,
                linestyle="dashed",
            )

            # if "robocrowd" not in dataset:
            min_quality, max_quality = np.min(quality_labels), np.max(quality_labels)
            spacing_factor = 0.1 if dataset in DATASET_ORDERS["robocrowd_image"] else 0.2
            ax.set_ylim(np.mean(quality_labels) - spacing_factor * (max_quality - min_quality), max_quality + 0.01)
            ax.set_xlim(np.min(x), np.max(x))

            # Write the title and labels in.
            ax_title = DATASET_NAMES.get(dataset, " ".join([s.capitalize() for s in dataset.split("_")]))
            ax.set_title(ax_title, pad=1)
            sns.despine(ax=ax)
            ax.tick_params(axis="y", pad=-2, labelsize=5)
            ax.tick_params(axis="x", pad=-2, labelsize=5)
            if i == 0:
                ax.set_ylabel("Avg Quality Score", labelpad=0)
            else:
                ax.set_ylabel(None)
            ax.get_legend().remove()
            ax.set_xlabel("Num Ep. Filtered", labelpad=0.5)

    else:
        x, y = load_scores(path)
        x, y = convert_to_threshold_curve(x, y)
        fig, ax = plt.subplots(1, 1, figsize=(scale_factor, scale_factor))
        plot_curve(ax, [x], [y], x_key=x_key, y_key=y_key, label=None)

    rect = [0, 0, 1, 1]
    if FLAGS.legend:
        handles, labels = ax.get_legend_handles_labels()
        bbox_offset = -0.075
        rect_offset = 0.11
        ncols = len(handles)
        if len(datasets) <= 2 and len(handles) > 6:
            rect_offset *= 4
            ncols = 3
        elif (len(handles) >= 6 and len(datasets) <= 3) or (len(handles) >= 5 and len(datasets) <= 2):
            rect_offset *= 2
            ncols = math.ceil(ncols / 2)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncols=ncols,
            bbox_to_anchor=(0.5, bbox_offset / figsize[1]),
            frameon=False,
        )
        rect[1] += rect_offset / figsize[1]

    if FLAGS.title is not None:
        rect[3] -= 0.032
        fig.suptitle(FLAGS.title)

    plt.tight_layout(pad=0, rect=rect)
    left = 0.01 if len(datasets) == 3 else 0
    plt.subplots_adjust(left=fig.subplotpars.left + left, wspace=fig.subplotpars.wspace + 0.02)
    plt.savefig(FLAGS.output, dpi=300)
    plt.clf()


if __name__ == "__main__":
    app.run(main)
