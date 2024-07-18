# MNIST Binary Classifier
# In this notebook, we will implement a CNN classifier to classify the digits 0 and 1 from the MNIST dataset. The objective of this lesson is twofold:
#
# To build our first CNN classifier (binary).
# To demonstrate the importance of cross-entropy.
# To compare the speed of CPU vs GPU training.
# Let's start with the ususal imports.

# For comparison purposes, we will force Tensorflow to use CPU here
# Cmment out this cell if you want to use GPU (if available)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras import Model
from time import time

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Set the seeds for reproducibility
from numpy.random import seed
from tensorflow.random import set_seed
seed_value = 1234578790
seed(seed_value)
set_seed(seed_value)

# Dataset Loading
# We have already inspected the MNIST dataset. We are going to load it now since we are going to use it for training the classifier.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Dataset params
num_classes = 10
size = x_train.shape[1]

print('Train set:   ', len(y_train), 'samples')
print('Test set:    ', len(y_test), 'samples')
print('Sample dims: ', x_train.shape)

# Dataset Preprocessing
# In this example, we are going to train a binary classifier to classify the digits 0 and 1. Therefore, we have to remove all other digits (classes) from the dataset.

mask_train = np.logical_or(y_train == 0, y_train == 1)
x_train = x_train[mask_train, ...]
y_train = y_train[mask_train]

mask_test = np.logical_or(y_test == 0, y_test == 1)
x_test = x_test[mask_test, ...]
y_test = y_test[mask_test]

# Normalization
x_train = x_train/255
x_test = x_test/255

print('Train set:   ', len(y_train), 'samples')
print('Test set:    ', len(y_test), 'samples')
print('Sample dims: ', x_train.shape)

# Building the Classifier
# We are going to build a relatively simple convolutional neural network (CNN) for this task.

# inputs = Input(shape=(size, size, 1))
#
# net = Conv2D(16, kernel_size=(3, 3), activation="relu")(inputs)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Conv2D(32, kernel_size=(3, 3), activation="relu")(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Flatten()(net)
# outputs = Dense(1, activation="linear")(net)
#
# model = Model(inputs, outputs)
# model.summary()
# # This is an extremely simple model (for a usual classification task) yet it already contains several thousand of (trainable) parameters. However, it still far less than the 12.5k parameters we used in the previous lesson.
#
# # Plot the model
# tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
#
# # Training
# # Let's now compile and train the model. We will use the well-known MSE as our loss function.
# #
# # Note: MSE is not the suitable loss for classification task but it serves us here well for the demonstration purposes. We will learn how to design a classifier in a proper way later in this lesson ;-)
#
# epochs = 25
# batch_size = 128
#
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
#
# start = time()
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# print('Elapsed time', time() - start)
#
# # Let's now plot the history to see the evolution of the training.
#
#
# def plot_history(history):
#     h = history.history
#     epochs = range(len(h['loss']))
#
#     plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
#     plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
#     plt.legend(['Train', 'Validation'])
#     plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
#                                epochs, h['val_accuracy'], '.-')
#     plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
#     plt.legend(['Train', 'Validation'])
#
#     print('Train Acc     ', h['accuracy'][-1])
#     print('Validation Acc', h['val_accuracy'][-1])
#
#
# plot_history(history)
# plt.show()
#
# # Evaluation
# # As we did in the previous lesson, we are going to evaluate the trained model.
#
# # Compute the predictions on the test set
# y_pred = model.predict(x_test)
#
# print('True', y_test[0:5].flatten())
# print('Pred', y_pred[0:5].flatten())
# y_true = y_test.flatten()
# y_pred = y_pred.flatten() > 0.5
#
# # Overall accuracy
# num_samples = len(y_true)
# acc = np.sum(y_test == y_pred)/num_samples
#
# # Accuracy for digit 0
# mask = y_true == 0
# acc0 = np.sum(y_test[mask] == y_pred[mask])/np.sum(mask)
#
# # Accuracy for digit 1
# mask = y_true == 1
# acc1 = np.sum(y_test[mask] == y_pred[mask])/np.sum(mask)
#
# print('Overall acc', acc)
# print('Digit-0 acc', acc0)
# print('Digit-1 acc', acc1)
# # We now visualise some of the evaluation results.
#
# for ii in range(15):
#     idx = np.random.randint(0, len(y_pred))
#     plt.subplot(3,5,ii+1), plt.imshow(x_test[idx, ...], cmap='gray')
#     plt.title('True: ' + str(y_true[idx]) + ' | Pred: ' + str(int(y_pred[idx])))
# plt.show()

# Full MNIST Classification
# So far we have implemented a binary MNIST classifier that is able to classify handwritten zeros and ones. However, the MNIST dataset contains (obviously) more digits. How can we extend the classifier to account for all possible digits? Would a straightforward extension work? Let's see :-)
#
# First, we are going to load the data (and keep all the digits!) and build the exactly same CNN as before.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalization
x_train = x_train/255
x_test = x_test/255

print('Train set:   ', len(y_train), 'samples')
print('Test set:    ', len(y_test), 'samples')
print('Sample dims: ', x_train.shape)
inputs = Input(shape=(size, size, 1))

net = Conv2D(16, kernel_size=(3, 3), activation="relu")(inputs)
net = MaxPooling2D(pool_size=(2, 2))(net)
net = Conv2D(32, kernel_size=(3, 3), activation="relu")(net)
net = MaxPooling2D(pool_size=(2, 2))(net)
net = Flatten()(net)
outputs = Dense(1, activation="linear")(net)

model = Model(inputs, outputs)
model.summary()

# Tranining
# We train the network the same way we did before. We are going to use the MSE as the loss function and demonstrate why it is not a good idea to use it for classification purposes :-) This is also the reason why we cannot use the built-in accuracy metric anymore.
#
# Also note that the training now takes longer since we are using all the available data (we didn't filter out any samples).

epochs = 25
batch_size = 128

model.compile(loss="mse", optimizer="adam", metrics=["mse"])

start = time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
print('Elapsed time', time() - start)

h = history.history
epochs = range(len(h['loss']))

plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
plt.legend(['Train', 'Validation'])
plt.show()
# Evaluation
# No we are going to evaluate the classifier accuracy for all digits.


y_pred = model.predict(x_test)

print('True', y_test[100:105].flatten())
print('Pred', y_pred[100:105].flatten())
y_true = y_test.flatten()
y_pred = y_pred.flatten()
digits = range(0, 10)

for digit in digits:
    mask = y_true == digit
    # Count the true positives (the closest digit)
    tp = np.sum(np.abs(y_pred[mask] - digit) < 0.5)
    total = np.sum(mask)
    print('Digit-', digit, ' acc', tp / total)

print('y_true', y_true[mask])
print('y_pred', y_pred[mask])
# We see that the accuracy is actually quite bad and the predictions are not close enough to the ground truth. The reason is that we are using an inappropriate loss function for the classification task..

plt.plot(y_true[mask], '.-')
plt.plot(y_pred[mask], '.-')
plt.grid(True)
plt.legend(['y_true', 'y_pred'])

plt.show()