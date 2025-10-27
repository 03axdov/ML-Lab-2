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
EMBED_MODEL_PATH = "models/embed_model.keras"
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
        metrics=["accuracy"],
    )
    history = model_copy.fit(small_x, small_y, epochs=100, verbose=0)
    print(f"??Final overfit accuracy: {history.history['accuracy'][-1]:.4f}")


# ---------------- Supervised Contrastive Loss ----------------
class SupConLoss(tf.keras.losses.Loss):
    def __init__(self, temperature: float = 0.1, name: str = "supcon_loss"):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        z = tf.cast(y_pred, tf.float32)
        z = tf.math.l2_normalize(z, axis=1)
        sim = tf.matmul(z, z, transpose_b=True) / tf.cast(self.temperature, tf.float32)

        B = tf.shape(z)[0]
        eye = tf.eye(B, dtype=tf.bool)
        not_eye = tf.logical_not(eye)

        labels = tf.expand_dims(y_true, 1)
        pos_mask = tf.equal(labels, tf.transpose(labels))
        pos_mask = tf.logical_and(pos_mask, not_eye)

        sim = sim - tf.reduce_max(sim, axis=1, keepdims=True)
        exp_sim = tf.exp(sim) * tf.cast(not_eye, tf.float32)
        denom = tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-12

        pos_exp = tf.exp(sim) * tf.cast(pos_mask, tf.float32)
        pos_sum = tf.reduce_sum(pos_exp, axis=1) + 1e-12

        valid = tf.reduce_any(pos_mask, axis=1)
        log_prob = -tf.math.log(pos_sum / tf.squeeze(denom, axis=1))
        log_prob = tf.boolean_mask(log_prob, valid)
        # If no valid anchors exist in the batch, return 0 to avoid NaN
        return tf.cond(
            tf.size(log_prob) > 0,
            lambda: tf.reduce_mean(log_prob),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )


# ---------------- Augmentation: Go Board Dihedral ----------------
def _apply_dihedral_to_sample(frames):
    # frames: (T, H, W, C)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    do_h = tf.random.uniform(()) > 0.5
    do_v = tf.random.uniform(()) > 0.5

    def aug_one(f):
        g = tf.image.rot90(f, k)
        g = tf.cond(do_h, lambda: tf.image.flip_left_right(g), lambda: g)
        g = tf.cond(do_v, lambda: tf.image.flip_up_down(g), lambda: g)
        return g

    return tf.map_fn(aug_one, frames)


def augment_batch(frames, labels):
    aug_frames = tf.map_fn(_apply_dihedral_to_sample, frames)
    return aug_frames, labels


# ---------------- Training Entry ----------------
def main():
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    print("Loading dataset...")

    train_ds, val_ds, label_encoder = prepare_frames_dataset(
        shard_pattern="data/processed/games_shard_*.npz",
        batch_size=BATCH_SIZE,
        val_split=0.1,
        seed=42,
    )

    x_batch, y_batch = next(iter(train_ds))
    input_shape = x_batch.shape[1:]
    num_classes = len(label_encoder.classes_)
    print(f"Detected input shape: {input_shape}")
    print(f"Detected {num_classes} classes.")

    # Build or load base classifier
    if os.path.exists(MODEL_PATH):
        print(f"?? Found existing model at {MODEL_PATH}. Loading...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("??Model loaded successfully.")
    else:
        print("?? No existing model found. Building new model...")
        model = build_game_model(
            board_size=BOARD_SIZE,
            history_k=HISTORY_K,
            max_moves=MAX_MOVES,
            num_classes=num_classes,
        )
        model.summary()

        # Sanity checks
        test_overfit(model, train_ds)
        with tf.GradientTape() as tape:
            preds = model(x_batch[:4])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch[:4], preds)
        grads = tape.gradient(loss, model.trainable_weights)
        nonzero = sum(float(tf.reduce_sum(tf.abs(g)).numpy() > 0) for g in grads if g is not None)
        print(f"Non-zero gradient tensors: {nonzero}/{len([g for g in grads if g is not None])}")

    # Training wrapper exposing both predictions and embedding
    preds = model.get_layer("predictions").output
    emb = model.get_layer("embedding").output
    train_model = tf.keras.Model(inputs=model.input, outputs={"predictions": preds, "embedding": emb})

    # Prepare datasets
    train_fit = train_ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    train_fit = train_fit.map(lambda x, y: (x, {"predictions": y, "embedding": y}), num_parallel_calls=tf.data.AUTOTUNE)
    val_fit = val_ds.map(lambda x, y: (x, {"predictions": y, "embedding": y}), num_parallel_calls=tf.data.AUTOTUNE)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    train_model.compile(
        optimizer=optimizer,
        loss={
            "predictions": "sparse_categorical_crossentropy",
            "embedding": SupConLoss(temperature=0.1),
        },
        loss_weights={"predictions": 1.0, "embedding": 0.1},
        metrics={"predictions": ["accuracy"]},
    )

    # Optionally resume from checkpoint weights if available
    ckpt_path = "models/cls_weights_best.h5"
    if os.path.exists(ckpt_path):
        print(f"?? Found checkpoint at {ckpt_path}. Loading weights to continue training...")
        try:
            train_model.load_weights(ckpt_path)
            print("??Checkpoint weights loaded successfully.")
        except Exception as e:
            print("⚠️ Could not load checkpoint weights:", e)

    # Callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/cls_weights_best.h5",
        monitor="val_predictions_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_predictions_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_predictions_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

    # Train
    initial_epoch = 4
    target_epochs = 20
    if os.path.exists(MODEL_PATH) or os.path.exists(ckpt_path):
        print(f"Continuing training up to {target_epochs} epochs...")
    else:
        print(f"Training new model for {target_epochs} epochs...")

    history = train_model.fit(
        train_fit,
        validation_data=val_fit,
        epochs=target_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    # Save base classifier and embedding submodel
    model.save(MODEL_PATH)
    print(f"??Full classifier model saved to {MODEL_PATH}")

    try:
        embed_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("embedding").output)
        embed_model.save(EMBED_MODEL_PATH)
        print(f"??Embedding submodel saved to {EMBED_MODEL_PATH}")
    except Exception as e:
        print("?��? Could not extract embedding submodel:", e)


if __name__ == "__main__":
    main()
