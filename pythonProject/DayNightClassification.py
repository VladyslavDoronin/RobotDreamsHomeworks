# Day-Night Image Classification
# In this notebook, we will implement a DNN classifier to classify images taken by stationary cameras as day or night images. We will also establish a baseline to make sure our deep learning implementation outperforms a simple hand-picked feature-based classification.

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 6]
# Dataset Loading
# We will use the day-night image dataset that you can download from here.

# Get the different video sequences
folder = '/home/user/Documents/GitHub/RobotDreamsHomeworks/pythonProject/data/archive/DNIM'
sequences = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(folder, 'time_stamp'))]

# Dataset Visualization
# Let'sw now visualize some random data and the corresponding labels to see what we are dealing with here.

sequence = sequences[1]

with open(os.path.join(folder, 'time_stamp', sequence + '.txt'), 'r') as fid:
    data = fid.readlines()
data = [d.strip() for d in data]

fnames = [d.split(' ')[0] for d in data]
hours = [int(d.split(' ')[-2]) for d in data]

for cnt, idx in enumerate(np.random.randint(0, len(fnames), 6)):
    image = cv2.imread(os.path.join(folder, 'Image', sequence, fnames[idx]))
    plt.subplot(2, 3, cnt + 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(hours[idx]), plt.axis(False)

plt.show()

# Data Preprocessing
# Now we will prepare the data for the training. That means that we load all the images, downsample them to a more treatable size and prepare the binary annotations.

size = 64
images, labels = [], []

for sequence in tqdm(sequences):
    # Read annotation file
    with open(os.path.join(folder, 'time_stamp', sequence + '.txt'), 'r') as fid:
        data = fid.readlines()

    # Extract information
    data = [d.strip() for d in data]
    fnames = [d.split(' ')[0] for d in data]
    hours = [int(d.split(' ')[-2]) for d in data]

    # Prepare training data
    images = images + [cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(folder, 'Image', sequence, f)),
                                               (size, size)), cv2.COLOR_BGR2RGB) for f in fnames]
    labels = labels + hours

images = np.array(images) / 255
labels = np.array(labels)

print(images.shape)
print(labels.shape)
# Divide the data into day and night images
labels = np.array([1 if (l <= 6 or l >= 18) else 0 for l in labels])

# Are the two classes balanced?
print('Num day images:  ', np.sum(labels == 0))
print('Num night images:', np.sum(labels == 1))
# Let's now shuffle the data and visualize them one last time.

images, labels = shuffle(images, labels)
for cnt, idx in enumerate(np.random.randint(0, len(images), 6)):
    plt.subplot(2, 3, cnt + 1), plt.imshow(images[idx], vmin=0, vmax=1), plt.title(labels[idx])
plt.show()

# Build the Baseline
# Before we start training our neural network based model, we might need to establish a performance baseline. We do so to be able to tell whether the model is actually good or not.
#
# Since we are dealing with a day-night problem, the most straightforward approach is to use the mean color (or even luminance) to classify images as day vs night. So let's do it :-)

# Compute mean color values for each image
mu_day = [np.mean(i) for i, l in zip(images, labels) if l == 0]#Вычисляем среднее значение картинки
mu_night = [np.mean(i) for i, l in zip(images, labels) if l == 1]

# Visualize the histograms of the obtained mean colors. Are they well separated?
counts_day, bins_day = np.histogram(mu_day, bins=10)
counts_night, bins_night = np.histogram(mu_night, bins=10)

plt.bar(bins_day[0:-1], counts_day, width=0.025)
plt.bar(bins_night[0:-1], counts_night, width=0.025)
plt.legend(['Day', 'Night']), plt.xlabel('average color'), plt.grid(True)
plt.show()

# The histograms do not seem that well separated. Anyway, let's find the optimal threshold that maximizes the classification accuracy.
acc = []
thresholds = np.arange(0.01, 1, 0.01)

for th in thresholds:
    acc.append((np.sum(mu_day >= th) + np.sum(mu_night < th)) / len(labels))

plt.plot(thresholds, acc, '.-'), plt.grid(True)
plt.xlabel('threshold'), plt.ylabel('accuracy')
plt.show()

# Binary Classifier Based on Neural Networks
# Let's now build a NN-based classifier and let's see if we can beat the baseline.
inputs = Input(shape=(size, size, 3))

net = Flatten()(inputs)
net = Dense(16, activation='relu')(net)
outputs = Dense(1, activation='sigmoid')(net)

model = Model(inputs, outputs)
model.summary()

# Let's compile the model and start training.
epochs = 25
batch_size = 128

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
history = model.fit(images, labels, batch_size=batch_size, epochs=epochs)
h = history.history
epochs = range(len(h['loss']))

plt.subplot(121), plt.plot(epochs, h['loss'], '.-'), plt.grid(True), plt.xlabel('epoch'), plt.title('loss')
plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-'), plt.grid(True), plt.xlabel('epoch'), plt.title('accuracy')
plt.show()
y_true = labels.flatten()
y_pred = model.predict(images).flatten()

y_pred_ = y_pred > 0.5

# Overall accuracy
num_samples = len(y_true)
acc = np.sum(y_true == y_pred_)/num_samples

# Accuracy for digit 0
mask = y_true == 0
acc0 = np.sum(y_true[mask] == y_pred_[mask])/np.sum(mask)

# Accuracy for digit 1
mask = y_true == 1
acc1 = np.sum(y_true[mask] == y_pred_[mask])/np.sum(mask)

print('Overall acc', acc)
print('Day-0 acc', acc0)
print('Night-1 acc', acc1)
# Visualisation
for ii in range(15):
    idx = np.random.randint(0, len(y_pred))
    plt.subplot(3,5,ii+1), plt.imshow(images[idx, ...]), plt.axis(False)
    plt.title('True: ' + str(y_true[idx]) + ' | Pred: ' + str(int(y_pred_[idx])))
plt.show()

