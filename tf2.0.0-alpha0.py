# https://www.tensorflow.org/alpha/tutorials/generative/dcgan
import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2DTranspose, Conv2D, BatchNormalization, LeakyReLU, Dropout, Reshape, Flatten, Lambda
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


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


def load_dataset(batch_size, dataset_name='mnist'):
    def preprocess_fn(features):
        # convert to 0.0 ~ 1.0
        features['image'] = tf.image.convert_image_dtype(features['image'], dtype=tf.float32)

        # convert to -1.0 ~ 1.0
        features['image'] = (features['image'] - 0.5) * 2.0
        return features

    # will return [28, 28, 1] uint8 (0~255)
    mnist_all, info = tfds.load(name=dataset_name, split=tfds.Split.ALL, with_info=True)
    print(info)
    total_examples = info.splits.total_num_examples

    # full shuffle
    mnist_all = mnist_all.shuffle(total_examples + 1)
    mnist_all = mnist_all.map(lambda x: preprocess_fn(x))
    mnist_all = mnist_all.batch(batch_size)
    mnist_all = mnist_all.prefetch(tf.data.experimental.AUTOTUNE)
    return mnist_all


class Generator(Model):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        n_filter = 64
        n_kernel = 5

        self.model = Sequential()

        # [z_dim] => [4, 4, 256]
        self.model.add(Dense(4 * 4 * n_filter * 4, input_shape=(z_dim,)))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Reshape((4, 4, n_filter * 4)))

        # [4, 4, 256] => [7, 7, 128]
        self.model.add(Conv2DTranspose(n_filter * 2, n_kernel, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Lambda(lambda x: x[:, :7, :7, :]))

        # [7, 7, 128] => [14, 14, 64]
        self.model.add(Conv2DTranspose(n_filter, n_kernel, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        # [14, 14, 64] => [28, 28, 1]
        self.model.add(Conv2DTranspose(1, n_kernel, strides=2, padding='same', activation='tanh'))
        self.model.summary()

    def call(self, x, training):
        return self.model(x, training)


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        n_filter = 64
        n_kernel = 5

        self.model = Sequential()

        # [28, 28, 1] => [14, 14, 64]
        self.model.add(Conv2D(n_filter, n_kernel, strides=2, padding='same', input_shape=(28, 28, 1)))
        self.model.add(LeakyReLU())

        # [14, 14, 64] => [7, 7, 128]
        self.model.add(Conv2D(n_filter * 2, n_kernel, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        # [7, 7, 128] => [4, 4, 256]
        self.model.add(Conv2D(n_filter * 4, n_kernel, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        # [4, 4, 256] => [1]
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.summary()

    def call(self, x, training):
        return self.model(x, training)


def gan_loss(d_real_logits, d_fake_logits):
    cross_entropy = BinaryCrossentropy(from_logits=True)
    d_loss_real = cross_entropy(tf.ones_like(d_real_logits), d_real_logits)
    d_loss_fake = cross_entropy(tf.zeros_like(d_fake_logits), d_fake_logits)
    d_loss = d_loss_real + d_loss_fake

    g_loss = cross_entropy(tf.ones_like(d_fake_logits), d_fake_logits)
    return d_loss, g_loss


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(real_images, noise, generator, discriminator, g_optimizer, d_optimizer):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)

        d_real_logits = discriminator(real_images, training=True)
        d_fake_logits = discriminator(fake_images, training=True)

        d_loss, g_loss = gan_loss(d_real_logits, d_fake_logits)

    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    return


def main():
    # parameters
    z_dim = 128
    batch_size = 128
    epochs = 30
    learning_rate = 1e-4
    num_examples_to_generate = 16
    model_name = 'mnist'
    save_dir = os.path.join('./assets', model_name)
    ckpt_dir = os.path.join('./models', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # start building models
    generator = Generator(z_dim)
    discriminator = Discriminator()
    g_optimizer = Adam(learning_rate)
    d_optimizer = Adam(learning_rate)

    # setup saving locations (object based savings)
    ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
    ckpt = tf.train.Checkpoint(g_optimizer=g_optimizer,
                               d_optimizer=d_optimizer,
                               generator=generator,
                               discriminator=discriminator)

    # get dataset
    mnist_dataset = load_dataset(batch_size=batch_size)

    # We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, z_dim])

    # start training
    for epoch in range(epochs):
        start = time.time()

        for features in mnist_dataset:
            real_images = features['image']
            running_batch_size = real_images.shape[0]
            noise = tf.random.normal([running_batch_size, z_dim])

            train_step(real_images, noise, generator, discriminator, g_optimizer, d_optimizer)

        # Produce images
        generate_and_save_images(generator, epoch + 1, seed, save_dir)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt.save(file_prefix=ckpt_prefix)

    # generating / saving after the final epoch
    generate_and_save_images(generator, epochs, seed, save_dir)
    ckpt.save(file_prefix=ckpt_prefix)
    return


if __name__ == '__main__':
    main()
