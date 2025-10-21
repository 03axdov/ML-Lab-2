import os
import tensorflow as tf
from dataset_generator import prepare_frames_dataset


BOARD_SIZE = 19
HISTORY_K = 4
MAX_MOVES = 200


def residual_block(x, filters, kernel_size=3):
    y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    if x.shape[-1] != filters:
        x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.ReLU()(tf.keras.layers.Add()([x, y]))
    return out


def build_move_encoder(input_shape, cnn_filters=32, cnn_blocks=2, mlp_units=128):
    inp = tf.keras.Input(shape=input_shape)  # H, W, C
    x = inp
    for _ in range(cnn_blocks):
        x = residual_block(x, cnn_filters)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(mlp_units, activation='relu')(x)
    return tf.keras.Model(inp, x, name='per_move_encoder')


def build_game_model(board_size=19, history_k=4, max_moves=200, num_classes=10, cnn_filters=32, cnn_blocks=2, move_dim=128, rnn_units=128, embed_dim=128):
    C = 2 * history_k + 2
    seq_inp = tf.keras.Input(shape=(max_moves, board_size, board_size, C))
    per_move = build_move_encoder((board_size, board_size, C), cnn_filters, cnn_blocks, move_dim)
    x = tf.keras.layers.TimeDistributed(per_move)(seq_inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units, return_sequences=False))(x)
    game_vec = tf.keras.layers.Dense(embed_dim, activation='tanh', name='game_vec')(x)
    norm_vec = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name='norm')(game_vec)
    out = tf.keras.layers.Dense(num_classes, activation='softmax', name='cls')(norm_vec)
    model = tf.keras.Model(seq_inp, out)
    embed_model = tf.keras.Model(seq_inp, norm_vec, name='embed_model')
    return model, embed_model


def main():
    data_dir = "data/train_set"
    train_ds, test_ds, label_encoder = prepare_frames_dataset(
        data_dir, board_size=BOARD_SIZE, history_k=HISTORY_K, max_moves=MAX_MOVES
    )
    num_classes = len(label_encoder.classes_)
    model, embed_model = build_game_model(
        board_size=BOARD_SIZE,
        history_k=HISTORY_K,
        max_moves=MAX_MOVES,
        num_classes=num_classes,
        cnn_filters=32,
        cnn_blocks=2,
        move_dim=128,
        rnn_units=128,
        embed_dim=128,
    )

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_ds, validation_data=test_ds, epochs=10)

    os.makedirs("models", exist_ok=True)
    model.save("models/cls_model.keras")
    embed_model.save("models/embed_model.keras")


if __name__ == "__main__":
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    main()

