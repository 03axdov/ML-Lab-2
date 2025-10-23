import os
import tensorflow as tf
from dataset import prepare_frames_dataset
from model import build_game_model
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")


# ---------------- Config ----------------
BOARD_SIZE = 19
HISTORY_K = 3           # must match preprocessing!
MAX_MOVES = 120
BATCH_SIZE = 32


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
    train_ds, val_ds, label_encoder = prepare_frames_dataset(
        shard_pattern="data/processed/games_shard_*.npz",
        batch_size=BATCH_SIZE,
        val_split=0.1,
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

    model.fit(train_ds, validation_data=val_ds, epochs=10)

    os.makedirs("models", exist_ok=True)
    model.save("models/cls_model.keras")
    print("✅ Models saved.")


if __name__ == "__main__":
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    main()
