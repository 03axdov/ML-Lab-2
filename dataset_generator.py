import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_parser import *


def prepare_dataset(data_dir, max_len=2000, vocab_size=20000, test_split=0.2, cache_dir="cache", force_rebuild=False):
    """Create or load cached tf.data.Dataset objects for training and testing."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_paths = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "y_test": os.path.join(cache_dir, "y_test.npy"),
        "tokenizer": os.path.join(cache_dir, "tokenizer.pkl"),
        "label_encoder": os.path.join(cache_dir, "label_encoder.pkl"),
    }

    # --- 1️⃣ Check if cache exists ---
    if (not force_rebuild) and all(os.path.exists(p) for p in cache_paths.values()):
        print("✅ Loading cached dataset...")
        X_train = np.load(cache_paths["X_train"])
        X_test = np.load(cache_paths["X_test"])
        y_train = np.load(cache_paths["y_train"])
        y_test = np.load(cache_paths["y_test"])
        with open(cache_paths["tokenizer"], "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_paths["label_encoder"], "rb") as f:
            label_encoder = pickle.load(f)

    else:
        print("⚙️ No cached dataset found — building new one...")

        # --- 2️⃣ Load and preprocess data ---
        games, labels = load_all_games(data_dir)
        print(f"Loaded {len(games)} games across {len(set(labels))} labels.")

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)

        # Tokenize SGF text
        tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False, oov_token='<OOV>')
        tokenizer.fit_on_texts(games)
        X = tokenizer.texts_to_sequences(games)
        X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

        # Split train/test with stratification to ensure all labels
        # are represented proportionally in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, stratify=y, random_state=42, shuffle=True
        )

        # --- 3️⃣ Save to cache ---
        np.save(cache_paths["X_train"], X_train)
        np.save(cache_paths["X_test"], X_test)
        np.save(cache_paths["y_train"], y_train)
        np.save(cache_paths["y_test"], y_test)
        with open(cache_paths["tokenizer"], "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_paths["label_encoder"], "wb") as f:
            pickle.dump(label_encoder, f)

        print(f"✅ Dataset cached under '{cache_dir}/'")

    # --- 4️⃣ Convert to TensorFlow datasets ---
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    return train_ds, test_ds, tokenizer, label_encoder


def prepare_frames_dataset(
    data_dir,
    board_size: int = 19,
    history_k: int = 4,
    max_moves: int = 200,
    test_split: float = 0.2,
    batch_size: int = 16,
    shuffle_buffer: int = 1024,
):
    """Create tf.data.Dataset of per-move spatial frames for each game.

    Each sample: tensor of shape (max_moves, board_size, board_size, C) where
    C = 2*history_k + 2. Labels are integer-encoded player IDs from file headers.
    """
    games, labels = load_all_games(data_dir)

    # Encode labels (if None, encode as 'unknown')
    labels = [lab if lab is not None else "<UNK>" for lab in labels]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Stratified split indices
    from sklearn.model_selection import train_test_split as _tts

    idx = np.arange(len(games))
    train_idx, test_idx, y_train, y_test = _tts(
        idx, y, test_size=test_split, stratify=y, random_state=42, shuffle=True
    )

    def gen(subset_idx):
        for i in subset_idx:
            sgf = games[i]
            moves = extract_moves_from_sgf(sgf, board_size=board_size)
            frames = moves_to_frames(moves, board_size=board_size, history_k=history_k, max_moves=max_moves)
            yield frames, y[i]

    C = 2 * history_k + 2
    output_signature = (
        tf.TensorSpec(shape=(max_moves, board_size, board_size, C), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )

    train_ds = tf.data.Dataset.from_generator(lambda: gen(train_idx), output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(lambda: gen(test_idx), output_signature=output_signature)

    train_ds = train_ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, label_encoder


if __name__ == "__main__":
    DATA_DIR = "data/train_set"
    train_ds, test_ds, tokenizer, label_encoder = prepare_dataset(DATA_DIR)
    print(f"Loaded {len(label_encoder.classes_)} players.")

    for X_batch, y_batch in train_ds.take(1):
        print("X_batch shape:", X_batch.shape)
        print("y_batch shape:", y_batch.shape)
        print("Example token IDs:", X_batch[0][:50].numpy())  # first 50 tokens
        print("Example label ID:", y_batch[0].numpy())
