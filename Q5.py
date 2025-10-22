import os
import tensorflow as tf
import numpy as np
from dataset import prepare_frames_dataset


# ---------------- Config ----------------
BOARD_SIZE = 19
HISTORY_K = 3           # must match preprocessing!
MAX_MOVES = 120
BATCH_SIZE = 32


# ---------------- Residual CNN Encoder ----------------
def residual_block(x, filters, kernel_size=3):
    """Residual block without BatchNorm (stable on float32 dense data)."""
    y = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', use_bias=False)(y)
    if x.shape[-1] != filters:
        x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    out = tf.keras.layers.ReLU()(tf.keras.layers.Add()([x, y]))
    return out


def build_move_encoder(input_shape, cnn_filters=32, cnn_blocks=2, mlp_units=96):
    """Per-move CNN encoder for AlphaGo-style board planes."""
    inp = tf.keras.Input(shape=input_shape)  # (H, W, C)
    x = inp
    for _ in range(cnn_blocks):
        x = residual_block(x, cnn_filters)

    # Global pooling + scaling to stabilize variance
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: t * 5.0)(x)
    x = tf.keras.layers.Dense(mlp_units, activation='relu')(x)
    return tf.keras.Model(inp, x, name='per_move_encoder')


# ---------------- Full Game Model ----------------
def build_game_model(input_shape, num_classes, cnn_filters=32, cnn_blocks=2, move_dim=96, rnn_units=128, embed_dim=128):
    """Full temporal Go game model with CNN+GRU architecture."""
    seq_inp = tf.keras.Input(shape=input_shape)  # (max_moves, 19, 19, C)

    per_move = build_move_encoder(input_shape[1:], cnn_filters, cnn_blocks, move_dim)
    x = tf.keras.layers.TimeDistributed(per_move)(seq_inp)

    # Mask padded timesteps
    x = tf.keras.layers.Masking(mask_value=0.0)(x)

    # Bidirectional GRU summarization
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_units, return_sequences=False)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Embedding projection
    game_vec = tf.keras.layers.Dense(embed_dim, activation='tanh', name='game_vec')(x)
    norm_vec = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name='norm')(game_vec)

    # Classifier head
    out = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='cls')(norm_vec)

    model = tf.keras.Model(seq_inp, out)
    embed_model = tf.keras.Model(seq_inp, norm_vec, name='embed_model')
    return model, embed_model


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

        
    x_batch, y_batch = next(iter(train_ds))
    print(tf.reduce_min(x_batch).numpy(), tf.reduce_max(x_batch).numpy())

    # Build model
    model, embed_model = build_game_model(
        input_shape=input_shape,
        num_classes=num_classes,
    )

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )


    # --- Gradient sanity check ---
    with tf.GradientTape() as tape:
        preds = model(x_batch[:4])
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch[:4], preds)
    grads = tape.gradient(loss, model.trainable_weights)
    nonzero = sum(
        [float(tf.reduce_sum(tf.abs(g)).numpy() > 0) for g in grads if g is not None]
    )
    print(f"Non-zero gradient tensors: {nonzero}/{len([g for g in grads if g is not None])}")

    # --- Train ---
    model.fit(train_ds, epochs=10)

    os.makedirs("models", exist_ok=True)
    model.save("models/cls_model.keras")
    embed_model.save("models/embed_model.keras")
    print("✅ Models saved.")


if __name__ == "__main__":
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    main()
