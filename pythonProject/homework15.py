# HOMEWORK 15
# In this homework we will be working with the Fashion MNIST dataset. You will be given a classifier which suffers from considerable overfitting. Your objective will be to employ regularization techniques to mitigate the overfitting problem.
#
# Let's start with the usual imports.

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, BatchNormalization
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

# Dataset
# The MNIST fashgion dataset link was build by Zalando Reasearch tem consists of monochrome images of different type of clothing, namely:
#
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot
# It is also one of the Keras built-in datasets. Let's load the images and quickly inspect it.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = np.array(x_train)/255
x_test = np.array(x_test)/255

y_train = np.array(y_train)
y_test = np.array(y_test)
# Dataset params
num_classes = 10
size = x_train.shape[1]

print('Train set:   ', len(y_train), 'samples')
print('Test set:    ', len(y_test), 'samples')
print('Sample dims: ', x_train.shape)
# Let's visualise some random samples.

cnt = 1
for r in range(3):
    for c in range(6):
        idx = np.random.randint(len(x_train))
        plt.subplot(3,6,cnt)
        plt.imshow(x_train[idx, ...], cmap='gray')
        plt.title(y_train[idx])
        cnt = cnt + 1
plt.show()

# Building the Classifier
# We are now going to build the baseline classifier that you will use throughout this homework.
# Data normalization
# x_train = x_train / 255
# x_test = x_test / 255
# inputs = Input(shape=(28, 28, 1))
# net = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
# net = Flatten()(net)
# net = Dense(128)(net)
# outputs = Dense(10, activation="softmax")(net)
#
# model = Model(inputs, outputs)
# model.summary()
# epochs = 50
# batch_size = 64
#
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
#

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


# plot_history(history)
# plt.show()
# As you can see, the classifier suffers from massive overfitting. The validation accuracy is around 88% while the training accuracy is close to 1. Даже больше. У меня 1.303

# Combat the Overfitting!
# Now it is your turn. Use the classifier as a baseline, include some regularization techniques and try to improve the classification performance. You can try any techniques you might see fit, e.g.,
#
# Dropout
# Batch normalization
# Weight regularization
# Data augmentation
# Early stopping
# Pooling
# Reducing the number of parameters (the size of the network)
# ...
# There are to objective you shall fulfill in order to successfully complete this homework:
#
# The validation accuracy shall be above 91%
# Your network (with all the regularizations applied) shall not be larger than the baseline
from sklearn.utils import shuffle

def datagen(x, y, batch_size):
    num_samples = len(y)
    while True:
        for idx in range(0, num_samples, batch_size):
            x_ = x[idx:idx + batch_size, ...]
            y_ = y[idx:idx + batch_size]

            if len(y_) < batch_size:
                x, y = shuffle(x, y)
                break

            # Augmentation
            for idx_aug in range(batch_size):
                if np.random.rand() > 0.5: # Оставил так, хотя возможно надо 0.1 ставить так как у нас всего 10 классов
                    x_[idx_aug, ...] = np.fliplr(x_[idx_aug, ...]) # Зераклит картинку

            yield x_, y_


def augment_using_ops(images, labels):
	images = tf.image.random_flip_left_right(images)
	images = tf.image.random_flip_up_down(images)
	images = tf.image.rot90(images)
	return (images, labels)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
])
# Modify the baseline classifier in order to reduce the overfitting and make the performance more robust

# inputs = Input(shape=(28, 28, 1))
# net = data_augmentation(inputs)
# net = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Dropout(0.25)(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# # net = Dropout(0.1)(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# # net = MaxPooling2D(pool_size=(2, 2))(net)
# # net = Dropout(0.2)(net)
# # net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Flatten()(net)
# net = Dense(128)(net)
# # outputs = Dense(10, activation="softmax")(net)
# outputs = Dense(10, activation="sigmoid")(net)

inputs = Input(shape=(28, 28, 1))
net = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
net = MaxPooling2D(pool_size=(2, 2))(net)
net = Dropout(0.2)(net)
net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
net = Flatten()(net)
net = Dense(128)(net)
outputs = Dense(10, activation="softmax")(net)

model = Model(inputs, outputs)
model.summary()

# Train the network
epochs = 50
batch_size = 64
steps_per_epoch = len(y_train) // batch_size
generator = datagen(x_train, y_train, batch_size)
# generator = augment_using_ops(x_train, y_train)
print(x_train.shape)
#
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=(x_test, y_test))
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
#
# Show the results
plot_history(history)
plt.show()
# Questions
# What have you done in order to improve the performance?

# Тренировка 2. В первую очередь, я попробовал сделать похожее, что мы делали на лекции. Добавил слои макспулинг, дропаут и конволюцию. Так же как и на лекции отзеркалил картинки.
# Да, понимаю что отзеркаливание тут сыграет маленькую роль, так как одежда на картинках выглядит в основном симетрично, но так же есть и обувь смотрящая лишь в одну сторону и штаны с согнутыми коленями. Так что Думаю должно улучшить.
# Тем не менее, результат вышел довольно хороший. Ниже показал результат. Оверфитинг довольно сильно уменьшился. Довольно высокий val_accuracy=0.9217
# Была такая настройка
# inputs = Input(shape=(28, 28, 1))
# net = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Dropout(0.2)(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Flatten()(net)
# net = Dense(128)(net)
# outputs = Dense(10, activation="softmax")(net)

# Тренировка 3. Тут попробую добавить батч нормализацию, и немного модифицирую аугументацию(попробую испольщзовать встроенные в тенсорфлоу аугументации).
# Плюс уменьшу Дропаут и добавлю его еще после одного слоя
# Вышел ужасный результат. Оверфитинг остался и по графику он очень зубастый непредсказуеммый(скорее всего дело в дропауте) и по чуть чуть растет. Если бы взял 100 епох, думао привысил бы 1
# Тут следующая настройка
# nputs = Input(shape=(28, 28, 1))
# net = data_augmentation(inputs)
# net = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Dropout(0.1)(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Dropout(0.1)(net)
# net = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Flatten()(net)
# net = Dense(128)(net)
# outputs = Dense(10, activation="softmax")(net)

# Тренировка 4. Уберу из тренировки 3 дропауты и добавлю конволюций.
# Снова плохо рузультат. Сильно выражен оверфитинг.
# inputs = Input(shape=(28, 28, 1))
# net = data_augmentation(inputs)
# net = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Flatten()(net)
# net = Dense(128)(net)
# outputs = Dense(10, activation="softmax")(net)

# Тренировка 5. Попробую использовать активацию сигмоида и чуть уменьше фильтр в конволюциях из тренировки 4. и Добавлю 1 дропаут
# Результат тоже так себе. Присутствует оверфитинг. Пока лучши результат это моя первая попытка
# inputs = Input(shape=(28, 28, 1))
# net = data_augmentation(inputs)
# net = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Dropout(0.25)(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = BatchNormalization()(net)
# net = MaxPooling2D(pool_size=(2, 2))(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(net)
# net = Flatten()(net)
# net = Dense(128)(net)
# outputs = Dense(10, activation="sigmoid")(net)

# Have you tried configurations that did not work out?
# Провел 5 тренировок и каждая из них смогла немного улучшить результат. Думаю основным улучшением было скорее аугументация данных.
# Так что пока не конфигураций которые вообще не сработали.

# Results:
# 1. Default train
# accuracy: 0.9939 - loss: 0.0168 - val_accuracy: 0.8838 - val_loss: 1.3030
# Train Acc      0.9940166473388672
# Validation Acc 0.8838000297546387

# 2. Тренировка 2.
# accuracy: 0.9673 - loss: 0.0842 - val_accuracy: 0.9217 - val_loss: 0.3027
# Train Acc      0.9662820100784302
# Validation Acc 0.9217000007629395

# 3. Тренировка 3.
# accuracy: 0.9858 - loss: 0.0371 - val_accuracy: 0.9160 - val_loss: 0.5165
# Train Acc      0.9857500195503235
# Validation Acc 0.9160000085830688

# 4. Тренировка 4.
# accuracy: 0.9932 - loss: 0.0225 - val_accuracy: 0.9102 - val_loss: 0.7930
# Train Acc      0.9935666918754578
# Validation Acc 0.9101999998092651

# 5. Тренировка 5.
# accuracy: 0.9875 - loss: 0.0341 - val_accuracy: 0.9092 - val_loss: 0.5920
# Train Acc      0.987500011920929
# Validation Acc 0.9092000126838684

# Отсавил в коде лучшую конфигурацию которая вышла


