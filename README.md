# Tensorflow Eager on GAN


## Basic usage - official Tensorflow documents
* [Eager Documentation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager)
    * [User guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/g3doc/guide.md)
    * [Basic usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/1_basics.ipynb)
    * [Gradient usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/2_gradients.ipynb)
    * [Data usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/notebooks/3_datasets.ipynb)

* GAN implementation from official example
    * https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/gan

## How I did it

### Tensorflow version
* 1.6

### What needs to be done
* Tensorflow 1.6 doesn't have tfe.Checkpoint class to save variables and more. (master branch does)
    * Which makes difficult to save & load variables
    * Use tfe.Saver() & tfe.restore_variables_on_create() -> works but not handy.

### Simple tip
* Eager mode is alot like pytorch. If you are familiar with pytorch then it should be easy.
* In tf.layers module, functions that starts with lower case is compatible with tensorflow graph mode. While Upper case works with eager mode.
* Ex) 
```python
tf.layers.conv2d() # use with graph mode
tf.layers.Conv2d() # use with eager mode
```
* In tensorflow eager mode, you must specify to use GPU. Otherwise it uses CPU.
```python
# create generator & discriminator with GPU support
device = '/gpu:0'
with tf.device(device):
    generator = Generator()
    discriminator = Discriminator()
```

### tf_eager_on_gan_v1.py
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
    
### tf_eager_on_gan_v2.py
* Trying to use tf.Network class and tf.track_layer() function for ease variable accessing.
* The gradient calculation can be archieved with ```tfe.GradientTape(persistent=True)```

### tf_eager_on_gan_v3.py
* Trying to use tf.data.Dataset api
* note: tested on tensorflow 1.8
    * No need tf.device('/gpu:0') ??