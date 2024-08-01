# Semantic Segmentation
# Semantic segmentation is a pixel-wise classification problem. In this notebook we are going to demonstrate how to build and train a CNN model to perform the segmentation task. We are going to employ the Oxford pet dataset (link).

import os
import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
root = "/home/user/Documents/GitHub/"
input_dir = os.path.join(root, "images")
target_dir = os.path.join(root, "annotations/trimaps/")
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
input_img_paths = sorted([os.path.join(input_dir, fname) for fname in input_img_paths])

target_img_paths = [f for f in os.listdir(target_dir) if f.endswith(".png") and not f.startswith(".")]
target_img_paths = sorted([os.path.join(target_dir, f) for f in target_img_paths])

assert len(input_img_paths) == len(target_img_paths)
print("Number of samples:", len(input_img_paths))

# Data Exploration
# Let's quickly explore the data to see what we are dealing with.

idx = 1000

img = cv2.cvtColor(cv2.imread(input_img_paths[idx]), cv2.COLOR_BGR2RGB)
mask = cv2.imread(target_img_paths[idx])

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(mask/255*50)
plt.show()

print(mask.shape)

# Model Definition
# We will use a variant of the famous U-Net architecture (paper). It is an encoder-decoder architecture suitable for solving segmentation tasks.

def get_model(img_size, num_classes):
    inputs = layers.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model


# Build model
model = get_model(img_size, num_classes)
model.summary()
plot_model(model, show_shapes=True)
plt.show()
# Data Generator
# Next, we are going to build a data generator so we can use it for training and validation.

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from sklearn.utils import shuffle

class OxfordPets(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to **batch** index."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1

        return x/255, y

    def on_epoch_end(self):
        self.input_img_paths, self.target_img_paths = shuffle(self.input_img_paths, self.target_img_paths)

# Training
import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)

# Split data
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data generators
train_gen = OxfordPets(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
callbacks = [keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)]

# Train the model, doing validation at the end of each epoch.
epochs = 15
# model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
# # Generate predictions for all images in the validation set
# val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
# val_preds = model.predict(val_gen)
#
# # Evaluation
# idx = 20
#
# mask = np.argmax(val_preds[idx], axis=-1)
# mask = np.expand_dims(mask, axis=-1)
#
# x = load_img(val_input_img_paths[idx])
# y_true = load_img(val_target_img_paths[idx])
# y_true = (np.array(y_true) - 1.0)/2
#
# rows, cols, _ = y_true.shape
# mask = cv2.resize(mask.astype(np.uint8), (cols, rows))
#
# plt.subplot(131), plt.imshow(x)
# plt.subplot(132), plt.imshow(y_true)
# plt.subplot(133), plt.imshow(mask, cmap='gray')
# plt.show()

