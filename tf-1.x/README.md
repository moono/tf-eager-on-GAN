# Tensorflow Eager on GAN

## Tensorflow 1.6
### Basic usage - official Tensorflow documents
* [Eager Documentation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager)
    * [User guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/g3doc/guide.md)
    * [Basic usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/1_basics.ipynb)
    * [Gradient usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/2_gradients.ipynb)
    * [Data usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/3_datasets.ipynb)

* GAN implementation from official example
    * https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/gan

* Tensorflow 1.6 doesn't have tfe.Checkpoint class to save variables and more. (master branch does)
    * Which makes difficult to save & load variables
    * Use tfe.Saver() & tfe.restore_variables_on_create() -> works but not handy.
* In tensorflow eager mode, you must specify to use GPU. Otherwise it uses CPU.
```python
# create generator & discriminator with GPU support
device = '/gpu:0'
with tf.device(device):
    generator = Generator()
    discriminator = Discriminator()
```

#### tf_eager_on_gan_v1.py
* Wrap Generator & Discriminator as class
* Define each layer with tf.layers Module
* Add generator & discriminator loss functions to calculate gradients
```python
# function that used to calculate gradient in the example is...
val_grad = tfe.implicit_value_and_gradients(some_loss_fn)
loss_ret, grads_vars = val_grad(...) # input parameters are same as 'some_loss_fn'
```
* Above function returns 'some_loss_fn' 's return value and associated gradients & variable pair(tuple).
* If we use above function to calculate the gradient and its variable, it returns all generator and discriminator variable.
* Which is not what we want.
* So I used ```tf.variable_scope()``` to find appropriate gradient and discriminator's gradient & variable pair. 
    
#### tf_eager_on_gan_v2.py
* Trying to use tf.Network class and tf.track_layer() function for ease variable accessing.
* The gradient calculation can be archieved with ```tfe.GradientTape(persistent=True)```

## Tensorflow 1.8
### tf_eager_on_gan_v3.py
* Trying to use tf.data.Dataset api
* No need tf.device('/gpu:0') ??

## Tensorflow 1.10
* tfe.Network class deprecated ??
* use tf.keras.Model
### tf_eager_on_gan_v4.py
* The code is from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb
* pix2pix model with tf.keras.Model