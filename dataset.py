# dataset_generator_sharded.py
import glob
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def load_all_shards(shard_pattern="data/processed/games_shard_*.npz"):
    """Generator that yields (frames, label) pairs from all NPZ shards."""
    shard_files = sorted(glob.glob(shard_pattern))
    if not shard_files:
        raise FileNotFoundError(f"No shards found matching {shard_pattern}")

    for path in shard_files:
        data = np.load(path)
        X, y = data["X"], data["y"]
        for frames, label in zip(X, y):
            yield frames, label


def prepare_frames_dataset(
    shard_pattern="data/processed/games_shard_*.npz",
    batch_size=32,
    shuffle_buffer=2048,
):
    """Creates a streaming tf.data.Dataset from NPZ shards."""
    # Load label encoder classes
    classes = np.load("data/processed/label_classes.npy", allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes

    # Infer one example shape from first shard
    first_shard = sorted(glob.glob(shard_pattern))[0]
    sample = np.load(first_shard)
    X_sample = sample["X"][0]
    C = X_sample.shape[-1]

    output_signature = (
        tf.TensorSpec(shape=(X_sample.shape), dtype=tf.float16),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: load_all_shards(shard_pattern),
        output_signature=output_signature,
    )

    # Cast to numerically stable dtypes for training
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, le
