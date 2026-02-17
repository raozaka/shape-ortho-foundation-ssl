import tensorflow as tf

def create_tf_dataset(images, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(images)

    if shuffle:
        ds = ds.shuffle(len(images))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
