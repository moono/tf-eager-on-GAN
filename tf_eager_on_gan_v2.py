import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange
from scipy.misc import toimage


def save_result(val_out, val_block_size, image_fn, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image, mode=color_mode).save(image_fn)


class Generator(tfe.Network):
    def __init__(self):
        super(Generator, self).__init__(name='generator')
        self.n_f = 512
        self.n_k = 4

        # input z vector is [None, 100]
        self.dense1 = self.track_layer(tf.layers.Dense(3 * 3 * self.n_f))
        self.conv2 = self.track_layer(tf.layers.Conv2DTranspose(self.n_f // 2, 3, 2, 'valid'))
        self.bn2 = self.track_layer(tf.layers.BatchNormalization())
        self.conv3 = self.track_layer(tf.layers.Conv2DTranspose(self.n_f // 4, self.n_k, 2, 'same'))
        self.bn3 = self.track_layer(tf.layers.BatchNormalization())
        self.conv4 = self.track_layer(tf.layers.Conv2DTranspose(1, self.n_k, 2, 'same'))
        return

    def call(self, inputs, is_trainig):
        with tf.variable_scope('generator'):
            x = tf.nn.leaky_relu(tf.reshape(self.dense1(inputs), shape=[-1, 3, 3, self.n_f]))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=is_trainig))
            x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=is_trainig))
            x = tf.tanh(self.conv4(x))
        return x


class Discriminator(tfe.Network):
    def __init__(self):
        super(Discriminator, self).__init__(name='discriminator')
        self.n_f = 64
        self.n_k = 4

        # input image is [-1, 28, 28, 1]
        self.conv1 = self.track_layer(tf.layers.Conv2D(self.n_f, self.n_k, 2, 'same'))
        self.conv2 = self.track_layer(tf.layers.Conv2D(self.n_f * 2, self.n_k, 2, 'same'))
        self.bn2 = self.track_layer(tf.layers.BatchNormalization())
        self.conv3 = self.track_layer(tf.layers.Conv2D(self.n_f * 4, self.n_k, 2, 'same'))
        self.bn3 = self.track_layer(tf.layers.BatchNormalization())
        self.flatten4 = self.track_layer(tf.layers.Flatten())
        self.dense4 = self.track_layer(tf.layers.Dense(1))
        return

    def call(self, inputs, is_trainig):
        with tf.variable_scope('discriminator'):
            x = tf.nn.leaky_relu(self.conv1(inputs))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=is_trainig))
            x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=is_trainig))
            x = self.dense4(self.flatten4(x))
        return x


# shorten sigmoid cross entropy loss calculation
def celoss_ones(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.ones_like(logits)*(1.0 - smooth)))


def celoss_zeros(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.zeros_like(logits)*(1.0 - smooth)))


def d_loss_fn(d_real_logits, d_fake_logits):
    d_loss_real = celoss_ones(d_real_logits, smooth=0.1)
    d_loss_fake = celoss_zeros(d_fake_logits, smooth=0.0)
    loss = d_loss_real + d_loss_fake
    return loss


def g_loss_fn(d_fake_logits):
    return celoss_ones(d_fake_logits, smooth=0.1)


def train(device):
    # hyper parameters
    z_dim = 100
    epochs = 30
    batch_size = 128
    learning_rate = 0.0002
    beta1 = 0.5
    is_training = True

    # for validation purpose
    assets_dir = './assets'
    if not os.path.isdir(assets_dir):
        os.makedirs(assets_dir)
    val_block_size = 10
    val_size = val_block_size * val_block_size

    # load mnist data
    mnist = input_data.read_data_sets('mnist-data', one_hot=True)
    inputs_shape = [-1, 28, 28, 1]

    # wrap with available device
    with tf.device(device):
        # create generator & discriminator
        generator = Generator()
        discriminator = Discriminator()

        # prepare optimizer
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        # for loss savings
        d_loss_at_steps = []
        g_loss_at_steps = []

        global_step = tf.train.get_or_create_global_step()

        for e in range(epochs):
            t = trange(mnist.train.num_examples // batch_size)
            # t = trange(1)
            for ii in t:
                t.set_description('{:04d}/{:04d}: '.format(e + 1, epochs))

                # no need labels
                batch_x, _ = mnist.train.next_batch(batch_size)

                # rescale images to -1 ~ 1
                batch_x = tf.reshape(batch_x, shape=inputs_shape)
                batch_x = batch_x * 2.0 - 1.0

                # Sample random noise for G
                batch_z = tf.random_uniform(shape=[batch_size, z_dim], minval=-1., maxval=1.)

                with tfe.GradientTape(persistent=True) as g:
                    # run generator first
                    fake_image = generator(batch_z, is_training)

                    # run discriminator
                    real_image = batch_x
                    d_real_logits = discriminator(real_image, is_training)
                    d_fake_logits = discriminator(fake_image, is_training)

                    # compute losses
                    d_loss = d_loss_fn(d_real_logits, d_fake_logits)
                    g_loss = g_loss_fn(d_fake_logits)

                # compute gradients
                d_grad = g.gradient(d_loss, discriminator.variables)
                g_grad = g.gradient(g_loss, generator.variables)

                # apply gradient via pre-defined optimizer
                d_optimizer.apply_gradients(zip(d_grad, discriminator.variables))
                g_optimizer.apply_gradients(zip(g_grad, generator.variables), global_step=global_step)

                # save loss
                d_loss_at_steps.append(np.asscalar(d_loss.numpy()))
                g_loss_at_steps.append(np.asscalar(g_loss.numpy()))

                # display current losses
                if ii % 5 == 0:
                    t.set_postfix(d_loss=d_loss.numpy(), g_loss=g_loss.numpy())

            # validation results at every epoch
            val_z = np.random.uniform(-1, 1, size=(val_size, z_dim))
            fake_image = generator(val_z, is_trainig=False)
            image_fn = os.path.join(assets_dir, 'gan-val-e{:03d}.png'.format(e + 1))
            save_result(fake_image.numpy(), val_block_size, image_fn, color_mode='L')

    return


def main():
    # Enable eager execution
    tfe.enable_eager_execution()

    # check gpu availability
    device = '/gpu:0'
    if tfe.num_gpus() <= 0:
        device = '/cpu:0'

    train(device)
    return


if __name__ == '__main__':
    main()
