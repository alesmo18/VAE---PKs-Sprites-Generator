# VAE-PKs-Sprites-Generator

## Introduction

This project deals with the implementation of CVAEs(Convolutional Variational Autoencoders) for [sprites](https://en.wikipedia.org/wiki/Sprite_(computer_graphics)) generation based on nintendo/game freaks' creatures.

In this project I'll test different types of newtorks to obtain a good looking generation by maximizing the accuracy of the model.

Currently I've created a basic model and modifying it slightly just to get 70% accuracy which is quite a low value, so this project has just begun and the results are still quite bad.

> Software and hardware used:

- Tensorflow: 2.5.0-rc3
- Cuda compilation tools, release 11.3, V11.3.58
- Build cuda_11.3.r11.3
- ipython 7.22.0
- jupyter core: 4.7.1
- jupyter-notebook: 6.3.0
- Intel(R) Core(TM) i5-10600K CPU @ 4.10GHz   4.10 GHz
- RAM: 16 gb
- NVIDIA GeForce RTX 3070


## Dataset creation

I've used sprites resized as 64x64, using 1 to 4 channels. Download source: [sprites source](https://veekun.com/dex/downloads)

This project uses a dataset composed of manually sampled images(a total of 12.158 imgs) which inlcude only the "front" size of a sprite, I've used sprites of Generation III, IV and V from the download source. These sprites have different size so I had to apply a scaling avoiding informations loss. I chose the 64x64 beacause it was the smallest one.

You can download the complete dataset here: [dataset](https://www.dropbox.com/s/z2iuwetzra7awkf/pk_sprites.rar?dl=0)

The main goal is to get a good accuracy using an input size of (64x64x4) so RGBA imgs, however the best I got actually is using grayscale sprites.

If you want to generate a grayscale sprites dataset just use the code:

```
directory = r'ORGINAL DATASET PATH'
out_dir = r'PATH WHERE TO SAVE NEWs GRAYSCALE SPRITES'
os.chdir(directory)
c=1
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        image_file = Image.open(filename) 
        image_file = image_file.convert('L') 
        im1 = np.asarray(image_file)
        im = Image.fromarray(im1)
        name='pksprite'+str(c)+'.png'
        im.save(out_dir+name) 
        c+=1
        print(os.path.join(directory, filename))
        continue
    else:
        continue
```

> All rights reserved on:
> Pokémon © 2002-2021 Pokémon. © 1995-2021 Nintendo/Creatures Inc./GAME FREAK inc. TM, ® and Pokémon character names are trademarks of Nintendo.
> No copyright or trademark infringement is intended in using Pokémon content.

## Load data

Libreries used on this project:
```
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential, Model, Input
from tensorflow.keras import losses
from PIL import Image
from tqdm import tqdm
```

I've used as input a numpy array of float32, load dataset code:
```
#LOADING THE DATA INTO A NUMPY ARRAY

dir_data = r'PUT HERE YOUR DATASET PATH'
pk_sprites = []
for img in tqdm(os.listdir(dir_data)):
    path = dir_data+'{}'.format(img)
    
    image = plt.imread(path)
    image = image.astype('float32')
    # removing transparancy png only on RGBA imgs
    # image = image[:, :, :3] 
    pk_sprites.append(image)
    
pk_sprites = np.array(pk_sprites)  
print(pk_sprites.shape)
```
Here's a simple print of a rgba sprite (using PIL lib)
![Sprite](https://i.ibb.co/x5fR8NR/Immagine.png)

> Be sure to change the variable of paths before use and check input shape of model


## Building the Convolutional Autoencoder Model

I've started checking on web a base model that would fit for 64x64 imgs and I slighty modified it.
To build the model I've used the tensorflow [API Keras Model Subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
The model is made of two Sequential part: encoder and decoder, each with 5 conv2d layers, 5 BatchNormalization layers, a Dense layer as encoder's output and a Dense as decoder's input. Input shape is 64x64x4, be sure to change it as 64x64x1 if you want to use grayscale sprites.

Base model code:
```
# Model base
latent_dims = 512

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(64, 64, 4)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=5,strides=1, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, strides=(2, 2), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=5, strides=(2, 2), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=5, strides=(2, 2), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.BatchNormalization(),
        ])

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=4*4*latent_dims, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Reshape(target_shape=(4, 4, latent_dims)),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=5, strides=(2, 2), padding='same',
                activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=5, strides=(2, 2), padding='same',
                activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=5, strides=(2, 2), padding='same',
                activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=5, strides=(2, 2), padding='same',
                activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=4, kernel_size=3, strides=1, padding='same',
                activation='sigmoid'),
            tf.keras.layers.BatchNormalization(),
        ])

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

vae = CVAE(latent_dims)

```
model.summary() result

![summary](https://i.ibb.co/2jb42QC/model-summary-kernel-size-5-latent-dim-512.png)

## Compile and fit

I've choose 30 epochs and a batch size of 64 for first trains, here is the code:
```
# COMPILE AND MODEL FIT
vae.compile(optimizer='adam', loss=losses.MeanSquaredError())
history = vae.fit(pk_sprites,pk_sprites, epochs=30, batch_size=64, verbose=1)
```
However if you're using 64x64x4 sprites, accuracy won't pass the 50%

## Results of first train

I've plotted a 10x10 grid of sprites: Originals, Reconstruced 

You can check some results here:

![Originals](https://i.ibb.co/FJKrdmg/originals.png)

![Reconstructed](https://i.ibb.co/H2HVsWF/reconstructed.png)

## Generation of news sprites

Finally I could generate the first bunch of new sprites, as code:

```
def plot_images(rows, cols, images, title):
    grid = np.zeros(shape=(rows*64, cols*64, 4))
    for row in range(rows):
        for col in range(cols):
            grid[row*64:(row+1)*64, col*64:(col+1)*64, :] = images[row*cols + col]

    plt.figure(figsize=(20,20))       
    plt.imshow(grid)
    plt.title(title)
    plt.show()
    
predictions= vae.decoder.predict(np.random.randn(100, latent_dims))
plot_images(10,10,predictions, "RANDOM SPRITES")
```
![New Sprites](https://i.ibb.co/zr8nZDX/new-sprites.png)

## Conclusion and future features

You can find the notebook python file uploaded, be sure to change paths before running, you can also find some plots results based on variations of base model.
In these tests I tryed different latent dimensions, batchs size, kernels size, where "w png hq4ch" stand for rgba imgs png format with 4 channels.
 
I'll keep to slighty modify the base model, next phase is to use the [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) to find the best params for the model
