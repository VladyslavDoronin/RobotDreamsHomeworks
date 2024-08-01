import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from CustomDataGenerator import CustomDataGenerator


root = "data"
input_dir_train = os.path.join(root, "MilVehicle/train")
target_dir_train = os.path.join(root, "Masks/mask_train")

input_dir_valid = os.path.join(root, "MilVehicle/valid")
target_dir_valid = os.path.join(root, "Masks/mask_valid")

input_dir_test = os.path.join(root, "MilVehicle/test")
target_dir_test = os.path.join(root, "Masks/mask_test")
batch_size = 16


# Загрузка модели
model = load_model('data/final_model.h5')

test_generator = CustomDataGenerator(images_path=input_dir_test, masks_path=target_dir_test, batch_size=batch_size)
test_preds = model.predict(test_generator)

idx = 21

mask = np.argmax(test_preds[idx], axis=-1)
mask = np.expand_dims(mask, axis=-1)

x = load_img(test_generator.image_files[idx])
y_true = load_img(test_generator.mask_files[idx])
y_true = (np.array(y_true) - 1.0)/2

rows, cols, _ = y_true.shape
mask = cv2.resize(mask.astype(np.uint8), (cols, rows))

plt.subplot(131), plt.imshow(x)
plt.subplot(132), plt.imshow(y_true)
plt.subplot(133), plt.imshow(mask, cmap='gray')
plt.show()