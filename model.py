import tensorflow as tf


# ----- Residual CNN Encoder (moderate size, no global pooling) -----
def build_residual_cnn_encoder(input_shape, filters=32, blocks=3, mlp_units=256):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu')(inp)
    for _ in range(blocks):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(mlp_units, activation='relu')(x)
    return tf.keras.Model(inp, x)


# ----- Game-level model -----
def build_game_model(board_size=19, history_k=3, max_moves=120, num_classes=200):
    """
    Returns a model with two outputs:
      - 'embedding': L2-normalized feature vector
      - 'predictions': softmax over num_classes
    During training, you can use only 'predictions' for loss,
    but at inference, use 'embedding' for few-shot matching.
    """
    C = 2 * history_k + 1
    seq_inp = tf.keras.Input(shape=(max_moves, board_size, board_size, C))

    # Encode each move board
    move_enc = build_residual_cnn_encoder((board_size, board_size, C))
    x = tf.keras.layers.TimeDistributed(move_enc)(seq_inp)

    # Use only the first N moves (optional truncation)
    x = tf.keras.layers.Lambda(lambda t: t[:, :30, :])(x)

    # Temporal modeling
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=False, dropout=0.2)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # Dense projection
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    # ---- Main embedding head ----
    emb = tf.keras.layers.Dense(128, activation='relu', name="embedding_pre")(x)
    emb = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="embedding")(emb)

    # ---- Classification head ----
    out = tf.keras.layers.Dense(num_classes, activation='softmax', name="predictions")(emb)

    # Build multi-output model (embedding + predictions)
    model = tf.keras.Model(inputs=seq_inp, outputs=out)

    return model
