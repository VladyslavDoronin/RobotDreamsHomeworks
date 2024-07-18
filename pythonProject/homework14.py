# HOMEWORK 14
# In this homework you are going to build your first classifier for the CIFAR-10 dataset. This dataset contains 10 different classes and you can learn more about it here. This homework consists of the following tasks:
#
# Dataset inspection
# Building the network
# Training
# Evaluation
# At the end, as usual, there will be a couple of questions for you to answer :-)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras import Model
from time import time

from matplotlib import pyplot as plt
from collections import Counter

plt.rcParams['figure.figsize'] = [15, 10]

# Set the seeds for reproducibility
from numpy.random import seed
from tensorflow.random import set_seed
seed_value = 1234578790
seed(seed_value)
set_seed(seed_value)

# Step 0: Dataset Inspection
# Load the dataset and make a quick inspection.

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Mapping from class ID to class name
classes = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer',
           5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# Dataset params
# num_classes = len(classes)
# print("len(classes): ", num_classes)
num_classes = len(Counter(y_train.flatten()))
print("Counter(y_train.flatten()): ", len(Counter(y_train.flatten())))
size = x_train.shape[1]

# Visualize random samples (as a plot with 3x6 samples)
for ii in range(18):
    plt.subplot(3,6,ii+1)
    # Pick a random sample
    idx = np.random.randint(0, num_classes)
    # Show the image and the label
    plt.imshow(x_train[idx, ...])
    plt.title(classes[int(y_train[idx])])

plt.show()
# Compute the class histogram (you can visualize it if you want). Is the dataset balanced?
# Hint: You might find Counter tool useful. In any case, it's up to you how you compute the histogram.
# print(Counter(y_train.flatten()))

hist = Counter(y_train.flatten())

plt.bar(hist.keys(), hist.values()), plt.grid(True)
plt.xticks(ticks=range(num_classes), labels=[classes[i] for i in range(10)])
plt.xlabel('Classes'), plt.ylabel('Counts')
plt.show()
# Dataset сбалансирован идеально. Всех классов равное количество по 5к
# Так же картинки все имеют одинаковый квадратный размер 30х30


# Step 1: Data Preparation
# In this step, you'll need to prepare the data for training, i.e., you will have to normalize it and encode the labels as one-hot vectors.

# Normalization
x_train = x_train/255
x_test = x_test/255

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print('Train set:   ', len(y_train), 'samples')
print('Test set:    ', len(y_test), 'samples')
print('Sample dims: ', x_train.shape)

# Step 2: Building the Classifier
# Build the CNN for CIFAR10 classification. For starters, you can use the same network we used in the lesson for the MNIST problem.

# Build the classifier
inputs = Input(shape=(size, size, 3))

net = Conv2D(16, kernel_size=(3, 3), activation="relu")(inputs)
# net = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
net = MaxPooling2D(pool_size=(2, 2))(net)
net = Conv2D(32, kernel_size=(3, 3), activation="relu")(net)
# net = Conv2D(16, kernel_size=(3, 3), activation="relu")(net)
# net = Conv2D(16, kernel_size=(3, 3), activation="relu")(net)
net = MaxPooling2D(pool_size=(2, 2))(net)
net = Flatten()(net)
# softmax это наша активация нейронов нормализируя рещультат чтоб все предсказания(сумма их) была равна 1. То есть есть 3 объекта разных, нейронка сказала шо это самолет на 90%(0.9), значит остальные два результата будут в пределх 10%(0.1=0.07+0.03)
outputs = Dense(num_classes, activation="softmax")(net)

model = Model(inputs, outputs)
# Show the model
model.summary()

# Step 3: Training
# Compile the model and train it.

# epochs = 25
epochs = 50
batch_size = 128

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model

start = time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
print('Elapsed time', time() - start)

# Show training history (this cell is complete, nothing to implement here :-) )
h = history.history
epochs = range(len(h['loss']))

plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
plt.legend(['Train', 'Validation'])
plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
                           epochs, h['val_accuracy'], '.-')
plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

print('Train Acc     ', h['accuracy'][-1])
print('Validation Acc', h['val_accuracy'][-1])

# Step 4: Evaluation
# In this step, you have to calculate the accuracies and visualize some random samples. For the evaluation, you are going to use the test split from the dataset.

# Compute the labels and the predictions as sparse values
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)

print('True', y_true[0:5])
print('Pred', np.argmax(model.predict(x_test), axis=1))
# print('Pred', y_pred[0:5, :])
print(y_pred.shape)

# Compute and print the accuracy for each class
for class_id, class_name in classes.items():
    mask = y_true == class_id
    tp = np.sum(y_pred[mask] == class_id)
    total = np.sum(mask)
    acc = tp/total
    print(class_name, acc)
    #plane 0.76
    # car 0.781
    # bird 0.526
    # cat 0.416
    # deer 0.754
    # dog 0.552
    # frog 0.757
    # horse 0.642
    # ship 0.662
    # truck 0.745

# Print the overall stats
ev = model.evaluate(x_test, y_test)
print('Test loss  ', ev[0])
print('Test metric', ev[1])
# Show random samples
for ii in range(15):
    # Pick a random sample
    idx = np.random.randint(0, len(y_pred))
    # Show the results
    plt.subplot(3, 5, ii+1), plt.imshow(x_test[idx, ...])
    plt.title('True: ' + str(classes[y_true[idx]]) + ' | Pred: ' + str(classes[y_pred[idx]]))
plt.show()
# Questions
# What is the overall accuracy of the classifier?
#  Если я правильно понял, то мы тут вывели общую точность классификатора print('Test metric', ev[1]). У меня вышло 0.6601999998092651. Но каждая новая тренировка получаю немного другие похожие числа



# What modifications would you do in order to improve the classification accuracy?
# Я думаю, я б в первую очередь повысил количество сверточных слоев(фильтров), например в Conv2D первый раз передал бы 32 и потом 64. Плюс повысил бы количество епох.
# Возможно нужно еще добавить дополнительные слои. Но с этом нужно играться и вычислять какие слои и куда еще лучше использовать



# Make one modification (that you think can help) and train the classifier again. Does the accuracy improve?
# Изменил количество епох до 50. Теперь Test metric 0.6712999939918518
# Изменил количество фильтров первой конволюции до 32. Теперь Test metric 0.6834999918937683
# Изменил количество фильтров второй конволюции до 128. Теперь Test metric 0.6520000100135803. Стало хуже. Возвращаю
# Добавил еще конволюций после второй конволюции несколько раз и разные пробовал. Улучшения нет