# import os
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from CustomDataGenerator import CustomDataGenerator
#
#
# root = "/home/user/Documents/data"
# input_dir_train = os.path.join(root, "MilVehicle/train")
# target_dir_train = os.path.join(root, "Masks/mask_train")
#
# input_dir_valid = os.path.join(root, "MilVehicle/valid")
# target_dir_valid = os.path.join(root, "Masks/mask_valid")
#
# input_dir_test = os.path.join(root, "MilVehicle/test")
# target_dir_test = os.path.join(root, "Masks/mask_test")
# batch_size = 8
#
# testType = "test"
# ANNOTATION_FILE_TEST = f'/home/user/Documents/data/MilVehicle/{testType}/_annotations.coco.json'
# class_names = ['btr', 'tank', 'truck']
# class_ids = [1, 2, 3]
#
# # Загрузка модели
# model = load_model('data/final_model.h5')
#
#
# test_generator_tank = CustomDataGenerator(images_path=input_dir_test, masks_path=target_dir_test, img_size=(128, 128),
#                                           annotation_file=ANNOTATION_FILE_TEST, class_id=class_ids[1],
#                                           batch_size=batch_size, shuffle=True)
# # Load the new image and preprocess it
# new_image = load_img("/home/user/Downloads/photo_2024-08-02_20-02-52.jpg", target_size=(128, 128))
# new_image_array = img_to_array(new_image) / 255.0  # Normalize the image
# test_preds = model.predict(new_image_array)
#
# idx = 11
#
# mask = np.argmax(test_preds[idx], axis=-1)
# mask = np.expand_dims(mask, axis=-1)
#
# print(test_generator_tank.image_files[idx])
# print(test_generator_tank.mask_files[idx])
# img = cv2.cvtColor(cv2.imread("/home/user/Downloads/photo_2024-08-02_20-02-52.jpg"), cv2.COLOR_BGR2RGB)
# maskTrue = cv2.imread(test_generator_tank.mask_files[idx])
#
# rows, cols, _ = maskTrue.shape
# mask = cv2.resize(mask.astype(np.uint8), (cols, rows))
#
# plt.subplot(131), plt.imshow(img)
# plt.subplot(132), plt.imshow(maskTrue/255)
# plt.subplot(133), plt.imshow(mask/255*50, cmap='gray')
# plt.show()
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the path to the new image
new_image_path = '/home/user/Downloads/photo_2024-08-02_20-02-52.jpg'

# Load the new image and preprocess it
new_image = load_img(new_image_path, target_size=(128, 128))
new_image_array = img_to_array(new_image) / 255.0  # Normalize the image

# Add batch dimension since model.predict expects a batch
new_image_array = np.expand_dims(new_image_array, axis=0)

# Load your model
# model = load_model('data/final_model.h5')
model = load_model('unetSegmentation.keras')

# Make prediction
new_image_pred = model.predict(new_image_array)

# Process the prediction to create a mask
new_image_mask = np.argmax(new_image_pred[0], axis=-1)
new_image_mask = np.expand_dims(new_image_mask, axis=-1)

# Load the original image and mask for visualization
original_img = cv2.cvtColor(cv2.imread(new_image_path), cv2.COLOR_BGR2RGB)

# Resize the mask to match the original image size
rows, cols, _ = original_img.shape
new_image_mask_resized = cv2.resize(new_image_mask.astype(np.uint8), (cols, rows))

# Visualize the results
plt.subplot(121), plt.imshow(original_img)
plt.subplot(122), plt.imshow(new_image_mask_resized, cmap='gray')
plt.show()

