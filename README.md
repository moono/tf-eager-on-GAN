# Tensorflow Eager on GAN

## Basic usage
* [Eager Documentation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager)
    * [User guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/g3doc/guide.md)
    * [Basic usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/1_basics.ipynb)
    * [Gradient usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/2_gradients.ipynb)
    * [Data usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/3_datasets.ipynb)

## GAN implementation from official example
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/gan

## How I did it
* Tensorflow version: 1.6

## What needs to be done
* Tensorflow 1.6 doesn't have tfe.Checkpoint class to save variables and more. (master branch does)
    * Which makes difficult to save & load variables
    * Use tfe.Saver() & tfe.restore_variables_on_create() -> works but not handy.

### Create classes for generator & discriminator
* Note: You can use tfe.Network class to subclass it and get the benefit of self.track_layer(), and Network.variables properties.
```python
# Generator class
class Generator(object):
    def __init__(self):
        self.n_f = 512
        self.n_k = 4

        # input z vector is [None, 100]
        self.dense1 = tf.layers.Dense(3 * 3 * self.n_f)
        self.conv2 = tf.layers.Conv2DTranspose(self.n_f // 2, 3, 2, 'valid')
        self.bn2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2DTranspose(self.n_f // 4, self.n_k, 2, 'same')
        self.bn3 = tf.layers.BatchNormalization()
        self.conv4 = tf.layers.Conv2DTranspose(1, self.n_k, 2, 'same')
        return

    def forward(self, inputs, is_trainig):
        with tf.variable_scope('generator'):
            x = tf.nn.leaky_relu(tf.reshape(self.dense1(inputs), shape=[-1, 3, 3, self.n_f]))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=is_trainig))
            x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=is_trainig))
            x = tf.tanh(self.conv4(x))
        return x

# Discriminator class
class Discriminator(object):
    def __init__(self):
        self.n_f = 64
        self.n_k = 4

        # input image is [-1, 28, 28, 1]
        self.conv1 = tf.layers.Conv2D(self.n_f, self.n_k, 2, 'same')
        self.conv2 = tf.layers.Conv2D(self.n_f * 2, self.n_k, 2, 'same')
        self.bn2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2D(self.n_f * 4, self.n_k, 2, 'same')
        self.bn3 = tf.layers.BatchNormalization()
        self.flatten4 = tf.layers.Flatten()
        self.dense4 = tf.layers.Dense(1)
        return

    def forward(self, inputs, is_trainig):
        with tf.variable_scope('discriminator'):
            x = tf.nn.leaky_relu(self.conv1(inputs))
            x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=is_trainig))
            x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=is_trainig))
            x = self.dense4(self.flatten4(x))
        return x

```

### Define loss function
* Note: All examples that I read had single loss function, but in this case(GAN) didn't worked. 
```python
# shorten sigmoid cross entropy loss calculation
def celoss_ones(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.ones_like(logits)*(1.0 - smooth)))


def celoss_zeros(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.zeros_like(logits)*(1.0 - smooth)))


# discriminator loss function
def d_loss_fn(generator, discriminator, input_noise, real_image, is_trainig):
    fake_image = generator.forward(input_noise, is_trainig)
    d_real_logits = discriminator.forward(real_image, is_trainig)
    d_fake_logits = discriminator.forward(fake_image, is_trainig)

    d_loss_real = celoss_ones(d_real_logits, smooth=0.1)
    d_loss_fake = celoss_zeros(d_fake_logits, smooth=0.0)
    loss = d_loss_real + d_loss_fake
    return loss


# generator loss function
def g_loss_fn(generator, discriminator, input_noise, is_trainig):
    fake_image = generator.forward(input_noise, is_trainig)
    d_fake_logits = discriminator.forward(fake_image, is_trainig)
    loss = celoss_ones(d_fake_logits, smooth=0.1)
    return loss
```

### Training
* Note: in tensorflow eager mode, you must specify to use GPU. Otherwise it uses CPU. 
```python
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
        d_val_grad = tfe.implicit_value_and_gradients(d_loss_fn)
        g_val_grad = tfe.implicit_value_and_gradients(g_loss_fn)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        # for loss savings
        d_loss_at_steps = []
        g_loss_at_steps = []

        for e in range(epochs):
            t = trange(mnist.train.num_examples // batch_size)
            # t = trange(1)
            for ii in t:
                t.set_description("{:04d}/{:04d}: ".format(e + 1, epochs))

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
                g_optimizer.apply_gradients(g_vars)

                # display current losses
                if ii % 5 == 0:
                    t.set_postfix(d_loss=d_loss_val.numpy(), g_loss=g_loss_val.numpy())

            # validation results at every epoch
            val_z = np.random.uniform(-1, 1, size=(val_size, z_dim))
            fake_image = generator.forward(val_z, is_trainig=False)
            image_fn = os.path.join(assets_dir, 'gan-val-e{:03d}.png'.format(e + 1))
            save_result(fake_image.numpy(), val_block_size, image_fn, color_mode='L')
    return
```

### Main function to start training
```python
def main():
    # Enable eager execution
    tfe.enable_eager_execution()

    # check gpu availability
    device = '/gpu:0'
    if tfe.num_gpus() <= 0:
        device = '/cpu:0'

    train(device)
    return
```