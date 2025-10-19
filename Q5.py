import tensorflow as tf
from dataset_generator import *


DATA_DIR = "data/train_set"


def main():
    # Force rebuild to ensure per-game samples after parser changes
    train_ds, test_ds, tokenizer, label_encoder = prepare_dataset(DATA_DIR)

    vocab_size = tokenizer.num_words
    embed_dim = 128
    num_classes = len(label_encoder.classes_)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(train_ds, validation_data=test_ds, epochs=10)


if __name__ == "__main__":
    main()
