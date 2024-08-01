import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from CustomDataGenerator import CustomDataGenerator
from tensorflow.keras.preprocessing.image import load_img

plt.rcParams['figure.figsize'] = [15, 10]

testType = "test"
ANNOTATION_FILE_TEST = f'../../../data/MilVehicle/{testType}/_annotations.coco.json'

root = "data"
input_dir_test = os.path.join(root, "MilVehicle/test")
target_dir_test = os.path.join(root, "Masks/mask_test")

class_names = ['btr', 'tank', 'truck']
class_ids = [1, 2, 3]
batch_size = 16

test_generator_tank = CustomDataGenerator(images_path=input_dir_test, masks_path=target_dir_test, img_size=(128, 128),
                                          annotation_file=ANNOTATION_FILE_TEST, class_id=class_ids[1],
                                          batch_size=batch_size, shuffle=True)

idx = 2

print(test_generator_tank.image_files[idx])
print(test_generator_tank.mask_files[idx])
img = cv2.cvtColor(cv2.imread(test_generator_tank.image_files[idx]), cv2.COLOR_BGR2RGB)
mask = cv2.imread(test_generator_tank.mask_files[idx])
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(mask/255)
plt.show()
