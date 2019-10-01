from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
# change  image to tensorflow shape, channel last ,dtype as float32
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
pic_255 = train_images[0].squeeze()

# normalizing the image to the range of [0., 1.]
train_images = train_images / 255.
test_images = test_images / 255.
pic_1 = train_images[0].squeeze()

# binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.
pic_b = train_images[0].squeeze()

plt.figure()
plt.subplot(131)
plt.imshow(pic_255, cmap='gray')
plt.axis('off')
plt.title(label='picture with 0-255')
plt.subplot(132)
plt.imshow(pic_1, cmap='gray')
plt.axis('off')
plt.title(label='picture with 0.-1.')
plt.subplot(133)
plt.imshow(pic_b, cmap='gray')
plt.title(label='picture change into binary.')
plt.axis('off')
plt.show()
