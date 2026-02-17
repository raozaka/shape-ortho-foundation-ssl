import tensorflow as tf

def build_encoder(input_shape=(128,128,1), embedding_dim=128):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = tf.keras.layers.Dense(embedding_dim)(x)

    return tf.keras.Model(inputs, outputs, name="encoder")
