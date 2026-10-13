import pickle

import numpy as np
import tensorflow as tf


def filter_by_ep_path(search_strings, path_key: str = "file_path"):
    def _filter(ep):
        string = ep["episode_metadata"][path_key]
        if isinstance(search_strings, list):
            bool_tensor = tf.concat(
                [tf.strings.regex_full_match(string, pattern=".*" + s + ".*") for s in search_strings], axis=0
            )
            return tf.math.reduce_all(bool_tensor)
        return tf.strings.regex_full_match(string, pattern=".*" + search_strings + ".*")

    return _filter


def filter_by_scores(score_path, attr, percentile=None, key=None):
    # Is there a nice way to XOR in python?
    assert percentile is None or key is None
    assert percentile is not None or key is not None

    with tf.io.gfile.GFile(score_path, "rb") as f:
        scores = pickle.load(f)[attr]

    if percentile is not None:
        threshold = np.percentile(np.array(list(scores.values())), percentile)
        # Keep all scores above a certain percentile
        keys = [k for k, v in scores.items() if v >= threshold]
    else:
        # Keep only a specific key.
        keys = [key]

    # A tf lookup that returns true if its in the set of keys, otherwise false.
    binary_set_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),  # Infer dtype automaticially
            values=tf.constant(tf.ones(len(keys), dtype=tf.int32)),
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )

    def _filter(ep):
        return tf.cast(binary_set_lookup.lookup(ep["episode_metadata"][attr]), tf.bool)

    return _filter


def quality_filter(threshold):
    def _filter(ep):
        return tf.cast(ep["episode_metadata"]["quality_score"] >= threshold, tf.bool)

    return _filter
