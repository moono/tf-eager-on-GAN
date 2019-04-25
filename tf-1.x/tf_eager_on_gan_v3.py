import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from scipy.misc import toimage


def make_generator(images, z_vectors):
    def _generator():
        for image, z_vec in zip(images, z_vectors):
            yield image, z_vec
    return _generator


def parse_fn(images, z_vec):
    # handle image
    # reshape to 28x28x1
    batch_x = tf.reshape(images, shape=[28, 28, 1])

    # rescale images to -1 ~ 1
    batch_x = batch_x * 2.0 - 1.0

    # create z vector
    batch_z = z_vec
    return batch_x, batch_z


def input_fn(mnist_images, train_z_vectors, batch_size):
    dataset = tf.data.Dataset.from_generator(make_generator(mnist_images, train_z_vectors), (tf.float32, tf.float32))
    dataset = dataset.map(parse_fn)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.shuffle(buffer_size=10000)  # randomize
    dataset = dataset.batch(batch_size)
    return dataset


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

    def call(self, inputs, is_training):
        with tf.variable_scope('generator'):
            x = tf.nn.leaky_relu(tf.reshape(self.dense1(inputs), shape=[-1, 3, 3, self.n_f]))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=is_training))
            x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=is_training))
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

    def call(self, inputs, is_training):
        with tf.variable_scope('discriminator'):
            x = tf.nn.leaky_relu(self.conv1(inputs))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=is_training))
            x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=is_training))
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


def grad_fn(generator, discriminator, input_images, z_vectors, is_training):
    with tfe.GradientTape(persistent=True) as g:
        # run generator first
        fake_image = generator(z_vectors, is_training)

        # run discriminator
        real_image = input_images
        d_real_logits = discriminator(real_image, is_training)
        d_fake_logits = discriminator(fake_image, is_training)

        # compute losses
        d_loss = d_loss_fn(d_real_logits, d_fake_logits)
        g_loss = g_loss_fn(d_fake_logits)

    # compute gradients
    d_grad = g.gradient(d_loss, discriminator.variables)
    g_grad = g.gradient(g_loss, generator.variables)

    return d_loss, g_loss, d_grad, g_grad


def run_generator(generator, z_dim, val_block_size):
    val_size = val_block_size * val_block_size

    # validation results at every epoch
    val_z = np.random.uniform(-1, 1, size=(val_size, z_dim))
    fake_image = generator(val_z, is_training=False)

    return fake_image


def train(generator, discriminator):
    # configure directories
    assets_dir = './assets'
    models_dir = './models'
    if not os.path.isdir(assets_dir):
        os.makedirs(assets_dir)
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    # hyper parameters
    params = {
        'z_dim': 100,
        'epochs': 30,
        'batch_size': 128,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'val_block_size': 10,
        'assets_dir': assets_dir,
        'models_dir': models_dir,
    }

    # load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_images = mnist.train.images  # Returns np.array
    train_z_vectors = np.random.uniform(-1.0, 1.0, size=(train_images.shape[0], params['z_dim']))

    # prepare saver
    checkpoint_directory = params['models_dir']
    checkpoint_prefix = os.path.join(checkpoint_directory, 'v3-ckpt')

    # prepare train data
    train_dataset = input_fn(train_images, train_z_vectors, params['batch_size'])

    # prepare optimizer
    d_optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=params['beta1'])
    g_optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=params['beta1'])

    # for loss savings
    d_losses = []
    g_losses = []

    # initiate saver (only need generator) - run dummy process first
    _ = generator(tf.zeros([1, params['z_dim']], dtype=tf.float32), is_training=False)
    generator_saver = tfe.Saver(var_list=generator.variables)

    # is_training flag
    is_training = True
    for e in range(params['epochs']):
        epoch_d_loss_avg = tfe.metrics.Mean()
        epoch_g_loss_avg = tfe.metrics.Mean()

        for mnist_images, z_vector in train_dataset:
            # Optimize the model
            d_loss, g_loss, d_grad, g_grad = grad_fn(generator, discriminator, mnist_images, z_vector, is_training)

            # apply gradient via pre-defined optimizer
            d_optimizer.apply_gradients(zip(d_grad, discriminator.variables))
            g_optimizer.apply_gradients(zip(g_grad, generator.variables),
                                        global_step=tf.train.get_or_create_global_step())

            # save loss
            epoch_d_loss_avg(d_loss)
            epoch_g_loss_avg(g_loss)

        d_losses.append(epoch_d_loss_avg.result())
        g_losses.append(epoch_g_loss_avg.result())

        print('Epoch {:03d}: d_loss: {:.3f}, g_loss: {:.3f}'.format(e + 1,
                                                                    epoch_d_loss_avg.result(),
                                                                    epoch_g_loss_avg.result()))

        # save every epoch's generator images
        fake_image = run_generator(generator, params['z_dim'], params['val_block_size'])
        image_fn = os.path.join(params['assets_dir'], 'gan-val-e{:03d}.png'.format(e + 1))
        save_result(fake_image.numpy(), params['val_block_size'], image_fn, color_mode='L')

        # save model
        generator_saver.save(checkpoint_prefix, global_step=tf.train.get_or_create_global_step())

    # visualize losses
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Losses')

    axes[0].set_ylabel('d_oss', fontsize=14)
    axes[0].plot(d_losses)
    axes[1].set_ylabel('g_loss', fontsize=14)
    axes[1].plot(g_losses)

    plt.show()
    return


def predict(generator):
    z_dim = 100

    # run dummy process
    _ = generator(tf.zeros([1, z_dim], dtype=tf.float32), is_training=False)

    # create test data
    val_block_size = 10
    val_size = val_block_size * val_block_size
    test_z = np.random.uniform(-1, 1, size=(val_size, z_dim))

    # initiate saver
    models_dir = './models'
    saver = tfe.Saver(var_list=generator.variables)
    saver.restore(tf.train.latest_checkpoint(models_dir))

    gen_out = generator(test_z, is_training=False)
    save_result(gen_out.numpy(), val_block_size, 'gen_out.png', color_mode='L')
    return


def main():
    # Enable eager execution
    tfe.enable_eager_execution()

    # create generator & discriminator
    generator = Generator()
    discriminator = Discriminator()

    # 1. train
    train(generator, discriminator)

    # 2. predict
    predict(generator)
    return


if __name__ == '__main__':
    main()
