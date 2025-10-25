import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow_addons.optimizers import AdamW
from dataset import prepare_frames_dataset
from model import build_game_model

# Use mixed precision globally
mixed_precision.set_global_policy("mixed_float16")

def stable_triplet_loss_v2(margin=0.2):
    """Triplet loss that never returns NaN, works with mixed precision."""
    margin = tf.constant(margin, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Pairwise squared distances
        dot = tf.matmul(y_pred, y_pred, transpose_b=True)
        sq_norm = tf.linalg.diag_part(dot)
        dist = tf.expand_dims(sq_norm, 0) - 2.0 * dot + tf.expand_dims(sq_norm, 1)
        dist = tf.maximum(dist, 0.0)

        # Build masks
        equal = tf.equal(tf.expand_dims(y_true, 0), tf.expand_dims(y_true, 1))
        pos_mask = tf.cast(equal, tf.float32)
        neg_mask = 1.0 - pos_mask

        # For each anchor, hardest positive and easiest negative
        hardest_pos = tf.reduce_max(dist * pos_mask, axis=1)
        max_dist = tf.reduce_max(dist, axis=1, keepdims=True)
        masked_neg = dist + max_dist * pos_mask
        hardest_neg = tf.reduce_min(masked_neg, axis=1)

        triplet = hardest_pos - hardest_neg + margin
        triplet = tf.maximum(triplet, 0.0)

        # Replace NaN with 0
        triplet = tf.where(tf.math.is_finite(triplet), triplet, tf.zeros_like(triplet))
        return tf.reduce_mean(triplet)

    return loss_fn


# ---------------- Config ----------------
BOARD_SIZE = 19
HISTORY_K = 3
MAX_MOVES = 120
BATCH_SIZE = 32
EPOCHS = 15
MODEL_PATH = "models/cls_model.keras"
EMBED_PATH = "models/embed_model.keras"


# ---------------- Training Script ----------------
def main():
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))

    # --- Load dataset ---
    train_ds, val_ds, label_encoder = prepare_frames_dataset(
        shard_pattern="data/processed/games_shard_*.npz",
        batch_size=BATCH_SIZE,
        val_split=0.1,
        seed=42,
    )



    # --- Important: map labels to both outputs ---
    def add_dual_labels(x, y):
        return x, {"predictions": y, "embedding": y}

    train_ds = train_ds.map(add_dual_labels)
    val_ds = val_ds.map(add_dual_labels)

    # --- Inspect batch shapes ---
    x_batch, y_batch = next(iter(train_ds))
    input_shape = x_batch.shape[1:]
    num_classes = len(label_encoder.classes_)
    print(f"Detected input shape: {input_shape}")
    print(f"Detected {num_classes} classes.")
    print("Output heads:", y_batch.keys())

    # --- Build or load model ---
    if os.path.exists(MODEL_PATH):
        print("🧠 Loading existing model...")
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss,
                "safe_l2_normalize": lambda x: x,  # placeholder for Lambda layer
            },
        )
    else:
        print("🚀 Building new model...")
        model = build_game_model(
            board_size=BOARD_SIZE,
            history_k=HISTORY_K,
            max_moves=MAX_MOVES,
            num_classes=num_classes,
        )
        model.summary()

    # --- Compile ---
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4, clipnorm=1.0)
    losses = {
        "predictions": "sparse_categorical_crossentropy",
        "embedding": stable_triplet_loss_v2(margin=0.2),
    }
    loss_weights = {"predictions": 1.0, "embedding": 0.2}

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={"predictions": ["accuracy"]},
    )

    # --- Callbacks ---
    os.makedirs("models", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/cls_model_best.keras",
            monitor="val_predictions_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_predictions_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_predictions_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --- Train ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # --- Save ---
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    try:
        embed_model = tf.keras.Model(
            model.input, model.get_layer("embedding").output
        )
        embed_model.save(EMBED_PATH)
        print(f"✅ Embedding submodel saved to {EMBED_PATH}")
    except Exception as e:
        print("⚠️ Could not extract embedding submodel:", e)


if __name__ == "__main__":
    main()
