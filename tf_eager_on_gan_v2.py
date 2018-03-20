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


def d_loss_fn(generator, discriminator, input_noise, real_image, is_trainig):
    fake_image = generator(input_noise, is_trainig)
    d_real_logits = discriminator(real_image, is_trainig)
    d_fake_logits = discriminator(fake_image, is_trainig)

    d_loss_real = celoss_ones(d_real_logits, smooth=0.1)
    d_loss_fake = celoss_zeros(d_fake_logits, smooth=0.0)
    loss = d_loss_real + d_loss_fake
    return loss


def g_loss_fn(generator, discriminator, input_noise, is_trainig):
    fake_image = generator(input_noise, is_trainig)
    d_fake_logits = discriminator(fake_image, is_trainig)
    loss = celoss_ones(d_fake_logits, smooth=0.1)
    return loss


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

    # for checkpoint saving
    ckpt_dir = './ckpt'
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_prefix_g = os.path.join(ckpt_dir, 'gan-generator')
    ckpt_prefix_d = os.path.join(ckpt_dir, 'gan-discriminator')

    # load mnist data
    mnist = input_data.read_data_sets('mnist-data', one_hot=True)
    inputs_shape = [-1, 28, 28, 1]

    # wrap with available device
    with tf.device(device):
        # create generator & discriminator
        generator = Generator()
        discriminator = Discriminator()

        # prepare optimizer
        d_val_grad = tfe.implicit_value_and_gradients(d_loss_fn)
        g_val_grad = tfe.implicit_value_and_gradients(g_loss_fn)
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

                # get loss related values & (gradients & vars)
                d_loss_val, d_grad_vars = d_val_grad(generator, discriminator, batch_z, batch_x, is_training)
                g_loss_val, g_grad_vars = g_val_grad(generator, discriminator, batch_z, is_training)

                # get appropriate gradients & variable pairs
                d_vars = [(grad, var) for (grad, var) in d_grad_vars if var.name.startswith('discriminator')]
                g_vars = [(grad, var) for (grad, var) in g_grad_vars if var.name.startswith('generator')]

                # save loss
                d_loss_at_steps.append(np.asscalar(d_loss_val.numpy()))
                g_loss_at_steps.append(np.asscalar(g_loss_val.numpy()))

                # apply gradient via pre-defined optimizer
                d_optimizer.apply_gradients(d_vars)
                g_optimizer.apply_gradients(g_vars, global_step=global_step)

                # display current losses
                if ii % 5 == 0:
                    t.set_postfix(d_loss=d_loss_val.numpy(), g_loss=g_loss_val.numpy())

            # validation results at every epoch
            val_z = np.random.uniform(-1, 1, size=(val_size, z_dim))
            fake_image = generator(val_z, is_trainig=False)
            image_fn = os.path.join(assets_dir, 'gan-val-e{:03d}.png'.format(e + 1))
            save_result(fake_image.numpy(), val_block_size, image_fn, color_mode='L')

            # save variables
            g_variables = (generator.variables + g_optimizer.variables())
            d_variables = (discriminator.variables + d_optimizer.variables() + [global_step])
            tfe.Saver(d_variables).save(ckpt_prefix_d, global_step=global_step)
            tfe.Saver(g_variables).save(ckpt_prefix_g, global_step=global_step)

    return


def test(device):
    z_dim = 100
    learning_rate = 0.0002
    beta1 = 0.5
    val_block_size = 10
    val_size = val_block_size * val_block_size

    ckpt_dir = './ckpt'
    ckpt_prefix_g = os.path.join(ckpt_dir, 'gan-generator')
    ckpt_prefix_d = os.path.join(ckpt_dir, 'gan-discriminator')

    # wrap with available device
    with tf.device(device):
        # create generator & discriminator
        with tfe.restore_variables_on_create('{:s}-12870'.format(ckpt_prefix_g)):
            generator = Generator()
            g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

            # the inference must be held within with context manager in this case...
            val_z = tf.random_uniform(shape=[val_size, z_dim], minval=-1., maxval=1.)
            fake_image = generator(val_z, is_trainig=False)
            image_fn = 'gan-test-out.png'
            save_result(fake_image.numpy(), val_block_size, image_fn, color_mode='L')

        with tfe.restore_variables_on_create('{:s}-12870'.format(ckpt_prefix_d)):
            discriminator = Discriminator()
            d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
            global_step = tf.train.get_or_create_global_step()

        print(global_step.numpy())
    return


def main():
    # Enable eager execution
    tfe.enable_eager_execution()

    # (device, data_format) = ('/gpu:0', 'channels_first')
    # if tfe.num_gpus() <= 0:
    #     (device, data_format) = ('/cpu:0', 'channels_last')

    # check gpu availability
    device = '/gpu:0'
    if tfe.num_gpus() <= 0:
        device = '/cpu:0'

    train(device)
    # test(device)

    return


if __name__ == '__main__':
    main()
