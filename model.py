import tensorflow as tf


# ----- Residual CNN Encoder (with global pooling for compact per-move features) -----
def build_residual_cnn_encoder(input_shape, filters=48, blocks=3, mlp_units=256, dropout=0.1):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    for _ in range(blocks):
        shortcut = x
        y = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        x = tf.keras.layers.Add()([y, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
    # Compact spatial summary per move
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(mlp_units, activation='relu')(x)
    return tf.keras.Model(inp, x)


def _masked_mean_time(x, mask):
    # x: (B, T, D), mask: (B, T) bool/float
    mask = tf.cast(mask, x.dtype)
    mask = tf.expand_dims(mask, axis=-1)  # (B, T, 1)
    x_sum = tf.reduce_sum(x * mask, axis=1)
    denom = tf.reduce_sum(mask, axis=1) + 1e-6
    return x_sum / denom


# (removed: old _masked_attention_pool helper; use explicit layers in build)


# ----- Game-level model -----
def build_game_model(board_size=19, history_k=3, max_moves=120, num_classes=200, emb_dim=128, scale=16.0, use_attention=True):
    """
    Returns a classifier model with an internal L2-normalized 'embedding' layer
    suitable for few-shot matching (prototypes/nearest-neighbors).

    Improvements for few-shot:
      - GlobalAvgPool per move to reduce params and overfitting.
      - Masked mean over time steps (no hard truncation).
      - Cosine-style classifier (normalized weights, scaled logits).
    """
    C = 2 * history_k + 1
    seq_inp = tf.keras.Input(shape=(max_moves, board_size, board_size, C))

    # Per-move encoder to feature vector
    move_enc = build_residual_cnn_encoder((board_size, board_size, C))
    feats = tf.keras.layers.TimeDistributed(move_enc)(seq_inp)  # (B, T, D)

    # Build a padding mask by detecting all-zero frames
    frame_nonzero = tf.reduce_any(tf.not_equal(seq_inp, 0.0), axis=[2, 3, 4])  # (B, T)

    # Masked temporal aggregation (attention or mean)
    if use_attention:
        # Attention MLP over time
        att_h = tf.keras.layers.Dense(128, activation='tanh', name='attn_fc1')(feats)
        att_logits = tf.keras.layers.Dense(1, name='attn_fc2')(att_h)  # (B, T, 1)
        att_logits2d = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='attn_logits_squeeze')(att_logits)
        # mask padded steps
        masked_logits = tf.keras.layers.Lambda(
            lambda z: tf.where(tf.equal(z[1], False), tf.cast(-1e9, z[0].dtype), z[0]),
            name='attn_mask'
        )([att_logits2d, frame_nonzero])
        attn_soft = tf.keras.layers.Softmax(axis=1, name='attn_softmax')(masked_logits)  # (B, T)
        # If a sequence has no valid steps, fall back to uniform attention
        valid_counts = tf.keras.layers.Lambda(lambda m: tf.reduce_sum(tf.cast(m, tf.float32), axis=1, keepdims=True), name='attn_valid_count')(frame_nonzero)
        uniform = tf.keras.layers.Lambda(lambda a: tf.ones_like(a) / tf.cast(tf.shape(a)[1], a.dtype), name='attn_uniform')(attn_soft)
        attn = tf.keras.layers.Lambda(
            lambda z: tf.where(z[1] > 0.0, z[0], z[2]),
            name='attn_safe'
        )([attn_soft, valid_counts, uniform])
        attn = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=-1), name='attn_expand')(attn)
        x = tf.keras.layers.Multiply(name='attn_weight')([feats, attn])
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name='attn_pool')(x)
    else:
        x = tf.keras.layers.Lambda(lambda z: _masked_mean_time(z[0], z[1]))([feats, frame_nonzero])  # (B, D)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # ---- Main embedding head ----
    pre = tf.keras.layers.Dense(emb_dim, activation='relu', name="embedding_pre")(x)
    emb = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="embedding")(pre)

    # ---- Cosine-style classification head ----
    logits = tf.keras.layers.Dense(
        num_classes,
        use_bias=False,
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
        name="cosine_logits",
    )(emb)
    logits = tf.keras.layers.Lambda(lambda z: z * scale, name="scaled_logits")(logits)
    out = tf.keras.layers.Activation('softmax', name="predictions")(logits)

    model = tf.keras.Model(inputs=seq_inp, outputs=out)
    return model
