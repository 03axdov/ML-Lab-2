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
MODEL_PATH = "models/cls_model.keras"
EMBED_MODEL_PATH = "models/embed_model.keras"   # <-- new
LOG_DIR = "logs"


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
    history = model_copy.fit(small_x, small_y, epochs=100, verbose=0)
    print(f"✅ Final overfit accuracy: {history.history['accuracy'][-1]:.4f}")


# ---------------- Training Entry ----------------
def main():
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    print("Loading dataset...")

    train_ds, val_ds, label_encoder = prepare_frames_dataset(
        shard_pattern="data/processed/games_shard_*.npz",
        batch_size=BATCH_SIZE,
        val_split=0.1,
        seed=42
    )

    x_batch, y_batch = next(iter(train_ds))
    input_shape = x_batch.shape[1:]  # (max_moves, 19, 19, C)
    num_classes = len(label_encoder.classes_)
    print(f"Detected input shape: {input_shape}")
    print(f"Detected {num_classes} classes.")

    # ---------------- Load or Build Model ----------------
    if os.path.exists(MODEL_PATH):
        print(f"🧠 Found existing model at {MODEL_PATH}. Loading...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        print("🚀 No existing model found. Building new model...")
        model = build_game_model(
            board_size=BOARD_SIZE,
            history_k=HISTORY_K,
            max_moves=MAX_MOVES,
            num_classes=num_classes,
        )
        model.summary()

        # Step 1: overfit test
        test_overfit(model, train_ds)

        # Step 2: gradient sanity check
        with tf.GradientTape() as tape:
            preds = model(x_batch[:4])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch[:4], preds)
        grads = tape.gradient(loss, model.trainable_weights)
        nonzero = sum(float(tf.reduce_sum(tf.abs(g)).numpy() > 0) for g in grads if g is not None)
        print(f"Non-zero gradient tensors: {nonzero}/{len([g for g in grads if g is not None])}")

    # ---------------- Compile ----------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    # ---------------- Callbacks ----------------
    os.makedirs("models", exist_ok=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/cls_model_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb, tensorboard_cb]

    # ---------------- Continue Training ----------------
    initial_epoch = 0
    target_epochs = 20
    if os.path.exists(MODEL_PATH):
        print(f"Continuing training up to {target_epochs} epochs...")
    else:
        print(f"Training new model for {target_epochs} epochs...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=target_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    # ---------------- Save Models ----------------
    model.save(MODEL_PATH)
    print(f"✅ Full classifier model saved to {MODEL_PATH}")

    # --- Also export embedding submodel for few-shot ---
    try:
        embed_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("embedding").output
        )
        embed_model.save(EMBED_MODEL_PATH)
        print(f"✅ Embedding submodel saved to {EMBED_MODEL_PATH}")
    except Exception as e:
        print("⚠️ Could not extract embedding submodel:", e)


if __name__ == "__main__":
    main()
