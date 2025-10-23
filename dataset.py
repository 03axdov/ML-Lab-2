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
    val_split=0.1,
):
    """Creates a streaming tf.data.Dataset from NPZ shards, with optional 10% val split."""
    shard_files = sorted(glob.glob(shard_pattern))
    if not shard_files:
        raise FileNotFoundError(f"No shards found matching {shard_pattern}")

    # --- Split shard files ---
    n_val = max(1, int(len(shard_files) * val_split))
    val_files = shard_files[:n_val]
    train_files = shard_files[n_val:]

    print(f"Using {len(train_files)} train shards, {len(val_files)} val shards")

    def make_dataset(files):
        ds = tf.data.Dataset.from_generator(
            lambda: load_all_shards_from_list(files),
            output_signature=(
                tf.TensorSpec(shape=(X_sample.shape), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            ),
        )
        ds = ds.map(
            lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # Infer one example shape
    first_shard = shard_files[0]
    sample = np.load(first_shard)
    X_sample = sample["X"][0]

    # Build two datasets
    train_ds = make_dataset(train_files)
    val_ds = make_dataset(val_files)

    # Load label encoder
    classes = np.load("data/processed/label_classes.npy", allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes

    return train_ds, val_ds, le


def load_all_shards_from_list(shard_files):
    for path in shard_files:
        data = np.load(path)
        X, y = data["X"], data["y"]
        for frames, label in zip(X, y):
            yield frames, label
