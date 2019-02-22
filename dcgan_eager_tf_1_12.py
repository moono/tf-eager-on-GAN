# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb
import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


tf.enable_eager_execution()


def make_generator_model(z_dim=100):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(z_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),
                                                     logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def train_step(images, batch_size, z_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):
    # generating noise from a normal distribution
    noise = tf.random_normal([batch_size, z_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))


def load_dataset_n_preprocess(batch_size):
    def preprocess_fn(features):
        features['image'] = tf.image.convert_image_dtype(features['image'], dtype=tf.float32)
        features['image'] = (features['image'] - 0.5) * 2.0
        return features

    # will return [28, 28, 1] uint8 (0~255)
    mnist_train = tfds.load(name='mnist', split=tfds.Split.TRAIN)
    mnist_test, info = tfds.load(name='mnist', split=tfds.Split.TEST, with_info=True)
    print(info)

    mnist_train = mnist_train.map(lambda x: preprocess_fn(x))
    mnist_train = mnist_train.shuffle(1024).batch(batch_size)
    mnist_train = mnist_train.prefetch(tf.data.experimental.AUTOTUNE)

    mnist_test = mnist_test.map(lambda x: preprocess_fn(x))
    return mnist_train, mnist_test


def generate_and_save_images(model, epoch, test_input, save_path):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close(fig)


def main():
    z_dim = 100
    epochs = 50
    batch_size = 32
    num_examples_to_generate = 16
    save_dir = os.path.join('./assets', 'dcgan_mnist')
    checkpoint_dir = os.path.join('./models', 'dcgan_mnist')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # We'll re-use this random vector used to seed the generator so
    # it will be easier to see the improvement over time.
    random_vector_for_generation = tf.random_normal([num_examples_to_generate, z_dim])

    # start building models
    generator = make_generator_model(z_dim)
    discriminator = make_discriminator_model()

    generator_optimizer = tf.train.AdamOptimizer(1e-4)
    discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

    # setup saving locations (object based savings)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # prepare data
    mnist_train, mnist_test = load_dataset_n_preprocess(batch_size)

    # optimze to graph for speed-up
    defun_train_step = tf.contrib.eager.defun(train_step)

    # start training
    for epoch in range(epochs):
        start = time.time()

        # for features in mnist_train.take(1):
        for features in mnist_train:
            defun_train_step(features['image'], batch_size, z_dim, generator, discriminator, generator_optimizer, discriminator_optimizer)

        generate_and_save_images(generator, epoch + 1, random_vector_for_generation, save_dir)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # generating after the final epoch
    generate_and_save_images(generator, epochs, random_vector_for_generation, save_dir)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return


if __name__ == '__main__':
    main()
