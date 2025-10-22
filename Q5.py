import os
import tensorflow as tf
import numpy as np
from dataset import prepare_frames_dataset

# ---------------- Config ----------------
BOARD_SIZE = 19
HISTORY_K = 3           # must match preprocessing!
MAX_MOVES = 120
BATCH_SIZE = 32


# ---------------- Simplified CNN Encoder ----------------
def build_move_encoder(input_shape, cnn_filters=64, mlp_units=256):
    """
    Per-move CNN encoder without BatchNorm or Dropout — built to test pure learnability.
    Uses Conv + ReLU + Flatten to preserve full spatial info.
    """
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(cnn_filters, 3, padding='same', activation='relu')(inp)
    x = tf.keras.layers.Conv2D(cnn_filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(dtype='float32')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(mlp_units, activation='relu')(x)
    return tf.keras.Model(inp, x, name='per_move_encoder')


# ---------------- Game-level Model ----------------
def build_game_model(board_size=19, history_k=3, max_moves=120, num_classes=200):
    """
    Simplified GRU-based model for overfit testing.
    """
    C = 2 * history_k + 1  # = 7 for K=3
    seq_inp = tf.keras.Input(shape=(max_moves, board_size, board_size, C))

    # Encode each move (per-move CNN)
    per_move = build_move_encoder((board_size, board_size, C))
    x = tf.keras.layers.TimeDistributed(per_move)(seq_inp)

    # Temporal summarization
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=False))(x)

    # Dense classifier head
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(dtype='float32')(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(seq_inp, out)
    return model


# ---------------- Overfit Test ----------------
def test_overfit(model, train_ds):
    """Train on a tiny subset to verify learnability (should reach >0.95 acc)."""
    small_x, small_y = next(iter(train_ds))
    small_x, small_y = small_x[:32], small_y[:32]
    print("Overfit test batch:", small_x.shape, small_y.shape)

    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    model_copy.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model_copy.fit(
        small_x, small_y,
        epochs=100,
        verbose=0
    )

    final_acc = history.history["accuracy"][-1]
    print(f"✅ Final overfit accuracy: {final_acc:.4f}")


# ---------------- Training Entry ----------------
def main():
    print("Loading dataset...")
    train_ds, label_encoder = prepare_frames_dataset(
        shard_pattern="data/processed/games_shard_*.npz",
        batch_size=BATCH_SIZE,
    )

    # Peek one batch to infer shape
    x_batch, y_batch = next(iter(train_ds))
    input_shape = x_batch.shape[1:]  # (max_moves, 19, 19, C)
    print(f"Detected input shape: {input_shape}")

    num_classes = len(label_encoder.classes_)
    print(f"Detected {num_classes} classes.")

    # Build model
    model = build_game_model(
        board_size=BOARD_SIZE,
        history_k=HISTORY_K,
        max_moves=MAX_MOVES,
        num_classes=num_classes,
    )

    model.summary()

    # --- Step 1: overfit test ---
    test_overfit(model, train_ds)

    # --- Step 2: sanity check gradients ---
    with tf.GradientTape() as tape:
        preds = model(x_batch[:4])
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch[:4], preds)
    grads = tape.gradient(loss, model.trainable_weights)
    nonzero = sum(
        [float(tf.reduce_sum(tf.abs(g)).numpy() > 0) for g in grads if g is not None]
    )
    print(f"Non-zero gradient tensors: {nonzero}/{len([g for g in grads if g is not None])}")

    # --- Step 3: full training ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    model.fit(train_ds, epochs=10)

    os.makedirs("models", exist_ok=True)
    model.save("models/cls_model.keras")
    print("✅ Models saved.")


if __name__ == "__main__":
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    main()
