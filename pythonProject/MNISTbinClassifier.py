import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from time import time
from matplotlib import pyplot as plt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all messages except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# If you want to use GPU, remove the next line
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Set the seeds for reproducibility
from numpy.random import seed
from tensorflow.random import set_seed

seed_value = 1234578790
seed(seed_value)
set_seed(seed_value)

plt.rcParams['figure.figsize'] = [15, 10]

# Your model code here


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

print('Train set:   ', len(y_train), 'samples')
print('Test set:    ', len(y_test), 'samples')
print('Sample dims: ', x_train.shape)

# Building the Classifier
# We are going to build a relatively simple fully-connected DNN for this task.

inputs = Input(shape=(size, size, 1))

net = Flatten()(inputs)# Разворачивает нашу матрицу в одну строку, то что мы делали раньше вручную
net = Dense(16, activation='relu')(net)
outputs = Dense(1)(net)

model = Model(inputs, outputs)
model.summary()

# This is an extremely simple model (for a usualo classification task) yet it already contains several thousand of (trainable) parameters.
# Training
# Let's now compile and train the model. We will use the well-known MSE as our loss function.
#
# Note: MSE is not the suitable loss for classification task but it serves us here well for the demonstration purposes. We will learn how to design a classifier in a proper way in the next lesson ;-)
epochs = 20
batch_size = 128

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

start = time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
print('Elapsed time', time() - start)
# Let's now plot the history to see the evolution of the training.


def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
                               epochs, h['val_accuracy'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    print('Train Acc     ', h['accuracy'][-1])
    print('Validation Acc', h['val_accuracy'][-1])

plot_history(history)
plt.show()

# Evaluation
# From the history we see that the training performance is quite consistent with the validation (which is good, we will learn about the overfitting problem in next lessons). Now we are going to evaluate the trained classifier on the test dataset. Remember, this is the dataset that the network has not seen during the training and it will be used to assess the final performance of the model.

y_pred = model.predict(x_test)

print('True', y_test[0:5].flatten())
print('Pred', y_pred[0:5].flatten())
y_true = y_test.flatten()
y_pred = y_pred.flatten()

y_pred_ = y_pred > 0.5

# Overall accuracy
num_samples = len(y_true)
acc = np.sum(y_test == y_pred_)/num_samples

# Accuracy for digit 0
mask = y_true == 0
acc0 = np.sum(y_test[mask] == y_pred_[mask])/np.sum(mask)

# Accuracy for digit 1
mask = y_true == 1
acc1 = np.sum(y_test[mask] == y_pred_[mask])/np.sum(mask)

print('Overall acc', acc)
print('Digit-0 acc', acc0)
print('Digit-1 acc', acc1)


# Precision & Recall
# Precision and recall describe the overall accuracy of the trained classifier (wiki).
precision, recall = [], []
thresholds = np.arange(0.1, 1, 0.1)

for th in thresholds:
    tp = np.sum(np.logical_and(y_test == 0, y_pred.flatten() < th))
    fp = np.sum(np.logical_and(y_test == 1, y_pred.flatten() < th))
    positives = np.sum(y_test == 0)

    recall.append(tp / positives)
    precision.append(tp / (tp + fp))

plt.plot(recall, precision, '.-'), plt.grid(True)
plt.xlabel('recall'), plt.ylabel('precision')
plt.show()
# We now visualise some of the evaluation results.
for ii in range(15):
    idx = np.random.randint(0, len(y_pred))
    plt.subplot(3, 5, ii + 1), plt.imshow(x_test[idx, ...], cmap='gray')
    plt.title('True: ' + str(y_true[idx]) + ' | Pred: ' + str(int(y_pred[idx])))
plt.show()

# Data Normalization
# The dynamic range of the input signals (images) is [0, 255] and the (groundtruth) output lies within the interval [0, 1]. This is a huge disproportion between the input and output ranges and the network needs to learn to compensate for that. This will slow the convergence and, in general, yields poorer results. In order to reduce the negative effect of the input-output range mismatch, data normalization is necessary. The objective of data normalization is to harmonize the input and output ranges and to let the network to focus on learning the important classification features instead of learning also the range compensation factors. A more advanced concept is the data standardisation which we will cover later.

# Data normalization
x_train = x_train/255
x_test = x_test/255
# Re-build the network to reinitilize the weights
inputs = Input(shape=(size, size, 1))
net = Flatten()(inputs)
net = Dense(16, activation='relu')(net)
outputs = Dense(1)(net)
model = Model(inputs, outputs)

# Compile and train
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
start = time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
print('Elapsed time', time() - start)

plot_history(history)
plt.show()

y_pred = model.predict(x_test)
y_pred = y_pred.flatten()

y_pred_ = y_pred > 0.5

# Overall accuracy
num_samples = len(y_true)
acc = np.sum(y_test == y_pred_) / num_samples

# Accuracy for digit 0
mask = y_true == 0
acc0 = np.sum(y_test[mask] == y_pred_[mask]) / np.sum(mask)

# Accuracy for digit 1
mask = y_true == 1
acc1 = np.sum(y_test[mask] == y_pred_[mask]) / np.sum(mask)

print('Overall acc', acc)
print('Digit-0 acc', acc0)
print('Digit-1 acc', acc1)
precision, recall = [], []
thresholds = np.arange(0.1, 1, 0.1)

for th in thresholds:
    tp = np.sum(np.logical_and(y_test == 0, y_pred.flatten() < th))
    fp = np.sum(np.logical_and(y_test == 1, y_pred.flatten() < th))
    positives = np.sum(y_test == 0)

    recall.append(tp / positives)
    precision.append(tp / (tp + fp))

plt.plot(recall, precision, '.-'), plt.grid(True)
plt.xlabel('recall'), plt.ylabel('precision')
plt.show()

# Visualisation
for ii in range(15):
    idx = np.random.randint(0, len(y_pred))
    plt.subplot(3, 5, ii + 1), plt.imshow(x_test[idx, ...], cmap='gray')
    plt.title('True: ' + str(y_true[idx]) + ' | Pred: ' + str(int(y_pred[idx])))

plt.show()


