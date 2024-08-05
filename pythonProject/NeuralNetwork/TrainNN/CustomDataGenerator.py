import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from pycocotools.coco import COCO

# Этот класс создан для получения всех необходимых картинок и масок для передачи в нейронку для обучения. А таже же для получения нужных картинок по индексу
class CustomDataGenerator(Sequence):
    def __init__(self, images_path, masks_path, img_size, annotation_file, class_id, batch_size=16, shuffle=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.img_size = img_size
        self.class_id = class_id
        self.batch_size = batch_size
        self.annotation_file = annotation_file
        self.coco = COCO(annotation_file)
        self.shuffle = shuffle
        self.image_ids = self.coco.getImgIds(catIds=class_id)
        self.image_files = [self.coco.loadImgs(img_id)[0]['file_name'] for img_id in self.image_ids]
        self.image_files = sorted([os.path.join(images_path, fname) for fname in self.image_files])
        self.mask_files = self.get_mask_files()
        self.mask_files = sorted([os.path.join(masks_path, fname) for fname in self.mask_files])

        print(len(self.image_files), ": ")
        print(self.image_files)
        print(len(self.mask_files), ": ")
        print(self.mask_files)

        self.indices = np.arange(len(self.image_files))

    def get_mask_files(self):
        all_masks = [f for f in os.listdir(self.masks_path) if f.endswith(".png") and not f.startswith("_")]
        mask_files = []
        for img_file in self.image_files:
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            corresponding_mask = [f for f in all_masks if os.path.splitext(f)[0] == img_name]
            if corresponding_mask:
                mask_files.append(os.path.join(self.masks_path, corresponding_mask[0]))
            else:
                mask_files.append(None)  # Or handle the case when the mask is missing
        return mask_files
    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_image_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_files = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self.__data_generation(batch_image_files, batch_mask_files)
        return images, masks


    def __data_generation(self, batch_image_files, batch_mask_files):
        images = []
        masks = []
        for img_file, mask_file in zip(batch_image_files, batch_mask_files):
            img_path = img_file
            mask_path = mask_file

            # mask_path = os.path.join(self.masks_path, mask_file)

            img = load_img(img_path, target_size=(128, 128))
            img = img_to_array(img) / 255.0
            images.append(img)

            mask = load_img(mask_path, target_size=(128, 128), color_mode="grayscale")
            mask = img_to_array(mask) / 255.0
            masks.append(mask)

        return np.array(images), np.array(masks)