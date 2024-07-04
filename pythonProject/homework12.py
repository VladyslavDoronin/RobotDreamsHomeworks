# In this homework, you are going to use the code from TrafficSignsClassification notebook and create your own traffic sign classifier.

# Step 1
# Use the data from data/subset_homework folder and visualize some examples. How many images are there for each class?
# Num samples class_0 2220
# Num samples class_1 2250

import os
import cv2
from time import time
import numpy as np
from sklearn.utils import shuffle

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 6]
# Data Loading
# Let's now load the data to see what we are dealing with.

folder = 'data/subset_homework'

# Load traffic sign class 0
fnames_0 = os.listdir(os.path.join(folder, 'class_id_0'))  # смотрим в папку class_id_0
images_0 = [cv2.imread(os.path.join(folder, 'class_id_0', f), cv2.IMREAD_UNCHANGED) for f in fnames_0] # читаем от туда все картинки
labels_0 = [0] * len(images_0)# для этого дорожного знака будет поментка(аутпут) = 0

for cnt, idx in enumerate(np.random.randint(0, len(images_0), 10)):
    plt.subplot(2,5,cnt+1)
    plt.imshow(images_0[idx], cmap='gray', vmin=0, vmax=255)
    plt.title(labels_0[idx]), plt.axis(False)

plt.show()

# Load traffic sign class 1
fnames_1 = os.listdir(os.path.join(folder, 'class_id_1'))
images_1 = [cv2.imread(os.path.join(folder, 'class_id_1', f), cv2.IMREAD_UNCHANGED) for f in fnames_1]
labels_1 = [1] * len(images_1)

for cnt, idx in enumerate(np.random.randint(0, len(images_1), 10)):
    plt.subplot(2,5,cnt+1)
    plt.imshow(images_1[idx], cmap='gray', vmin=0, vmax=255)
    plt.title(labels_1[idx]), plt.axis(False)

plt.show()

print('Num samples class_0', len(images_0))
print('Num samples class_1', len(images_1))
# Num samples class_0 2220
# Num samples class_1 2250

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------>
# Step 2
# Run the training with one single neuron (as we did in the lecture). What accuracy can you achieve?
# First time Accuracy 0.9259507829977629
# Second time Accuracy 0.9154362416107382
# Third time Accuracy 0.9331096196868008
# Fouth time Accuracy 0.9029082774049217

# Prepare Input Data
# To train our neural network model, we have to prepare the data to the format the the model actually expects. In our case, this will be numpy arrays.

# Put both classes together and shuffle the data
images = images_0 + images_1
labels = labels_0 + labels_1
images, labels = shuffle(images, labels)

images = np.array(images)
labels = np.array(labels)

print('Images', images.shape)
print('Labels', labels.shape)

# But now we have a problem. We cannot just feed the image to a neuron since the neuron inputs are flat (one dimensional). On the other hand, the images are 2D matrices. Therefore, we need to "flatten" the images to a one dimensional vector of pixels.
# Разварачиваем все наши картинки(каждую) в 1 большой вектоор чтоб использовать этот вектор как интпут к нашему нейрону
start = time()
pixels = []
for image in images:
    pixels_ = []
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            pixels_.append(image[r, c])
    pixels.append(pixels_)

pixels = np.array(pixels) / 255
stop = time()

print('Shape', pixels.shape)
print('Elapsed time', stop - start)

# Or, you know, just let make use of our friend numpy :-)

start = time()
#  flatten берет матрицу и разворачивает ее в 1 ветор и нормализирует к 255
pixels = np.array([image.flatten() for image in images])/255
stop = time()

print('Shape', pixels.shape)
print('Elapsed time', stop - start)
# Before the training, let's again have a look at some raqndom samples from our dataset.

for cnt, idx in enumerate(np.random.randint(0, len(images), 10)):
    plt.subplot(2,5,cnt+1)
    plt.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
    plt.title(labels[idx])

plt.show()



# Building the Neural Network
# Let's now build our first (and yes, very simple) neural network using Tensorflow. For that, we will need a couple of new imports.

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

# The following netowork will consist of only one single neuron. It is a very tiny network (not even a network, strictly speaking :-) ) and yet it can be quite powerful.

inputs = Input(shape=(pixels.shape[1],)) # первый слой этьо наш интпут. Говорим керасу какие размеры у нашего инпута ixels.shape[1] - это вектор который мы развернули из матрицы(нашей картинки)
outputs = Dense(1, activation="linear")(inputs) # колчесвто нейронов в одном слое. linear использовать для 1 слоя лучше, а для нескольких нужен рел. activation="linear" - это и так по умолчанию
model = Model(inputs, outputs)

# Before starting the training, we have to compile the model. During the compilation, we indicate what optimizer we want to use and what loss should be applied for the minimization process.

model.compile(optimizer ='adam', loss = 'mean_squared_error')
# And let's train :-)

history = model.fit(pixels, labels, epochs=10, batch_size=32) # тут начинается процесс обучения. 10 epochs значит что я хочу чтоб сеть прошла обучение 10 раз беря 32 картинки в рандломном порядке из нашей бд пиксельс в соответсвии с лебелс
# Plot training history
h = history.history
epochs = range(len(h['loss']))
plt.plot(epochs, h['loss'], '.-'), plt.grid(True)
plt.xlabel('epoch'), plt.ylabel('loss')
plt.show()


# Let's also have a looks at the learnt weights
plt.plot(model.layers[1].weights[0].numpy(), '.-'), plt.grid(True)
print(model.layers[1].weights[1].numpy(), model.layers[1].bias.numpy())
plt.show()

# Performance Evaluation
# Once our model is trained, we will can run it on our images to see how it performs (inference).

idx = 50
pred = model.predict(pixels[idx:idx+1, ...])
print(pred, labels[idx])
# Run it on the entire dataset
predictions = model.predict(pixels).squeeze()
predictions = predictions > 0.5
correct = 0
for prediction, label in zip(predictions, labels):
    if prediction == label:
        correct = correct + 1

print('Accuracy', correct/len(labels))
# First time Accuracy 0.9259507829977629
# Second time Accuracy 0.9154362416107382
# Third time Accuracy 0.9331096196868008
# Fouth time Accuracy 0.9029082774049217
for cnt, idx in enumerate(np.random.randint(0, len(images), 10)):
    plt.subplot(2,5,cnt+1), plt.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
    plt.title('Label: ' + str(labels[idx]) + ' | Prediction: ' + str(predictions[idx]))
    plt.axis(False)

plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------>
# Step 3
# Make further modifications to improve the accuracy (e.g. add more neurons, more layers, etc.). What is the maximum accuracy you can achieve?

from keras import activations
inputs = Input(shape=(pixels.shape[1],))
outputs = Dense(1, activation=activations.linear)(inputs)
# outputs = Dense(1, activation=activations.linear)(outputs)
# outputs = Dense(1, activation=activations.linear)(outputs)
model = Model(inputs, outputs)

model.compile(optimizer ='adam', loss = 'mean_squared_error')

# -----------
# history = model.fit(pixels, labels, epochs=10, batch_size=32)
# Простое повышение количества епох уже улучшает точность. Теперь она варируется от 0.96 до 0.978
# -----------

# -----------
# history = model.fit(pixels, labels, epochs=10, batch_size=128)
# Увеличение batch_size плохо влияет на Accuracy при маленьком количестве епох
# -----------

# -----------
history = model.fit(pixels, labels, epochs=100, batch_size=128)
# Тут уже с большим количеством епох чем было, batch_size уже не так плохо влияет на точность. Получаю акуранси такой же как и в первом случае - 0.96 до 0.97
# -----------

idx = 50
pred = model.predict(pixels[idx:idx+1, ...])
print(pred, labels[idx])
# Run it on the entire dataset
predictions = model.predict(pixels).squeeze()
predictions = predictions > 0.5
correct = 0
for prediction, label in zip(predictions, labels):
    if prediction == label:
        correct = correct + 1

print('Accuracy', correct/len(labels))


# -------------------------------------------------------------------------------------------------------------------->
# Пропую использовать другой активатор и другое количество нейронов
#

from tensorflow.keras.layers import Flatten

inputs = Input(shape=(pixels.shape[1],))
# inputs = Input(shape=(pixels.shape[1], pixels.shape[1], 3))
# outputs = Flatten()(inputs)
outputs = Dense(1, activation=activations.relu)(inputs)
# outputs = Dense(1, activation=activations.linear)(outputs)
model = Model(inputs, outputs)

model.compile(optimizer ='adam', loss = 'mean_squared_error')

history = model.fit(pixels, labels, epochs=10, batch_size=32)

idx = 50
pred = model.predict(pixels[idx:idx+1, ...])
print(pred, labels[idx])
# Run it on the entire dataset
predictions = model.predict(pixels).squeeze()
predictions = predictions > 0.5
correct = 0
for prediction, label in zip(predictions, labels):
    if prediction == label:
        correct = correct + 1

print('Accuracy', correct/len(labels))

# Изменил метод активации нейрона на relu и при epochs=10  batch_size=32 получаю такой же результат как для линера.
# Пробюовал изменить количество нейронов, но постоянно сталкивался с ошибками. Как понял там нужен Flatten и другие определенные размерност и это будем на следующих занятиях это рассматривать