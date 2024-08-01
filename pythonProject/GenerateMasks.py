import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

rootPath = "data/Masks/mask_"
trainType = "train"
valType = "valid"
testType = "test"
# Пути к датасету COCO
ANNOTATION_FILE_TRAIN = f'../../../data/MilVehicle/{trainType}/_annotations.coco.json'
ANNOTATION_FILE_VAL = f'../../../data/MilVehicle/{valType}/_annotations.coco.json'
ANNOTATION_FILE_TEST = f'../../../data/MilVehicle/{testType}/_annotations.coco.json'


def generate_masks(coco, output_dir, type):
    # Получение всех категорий
    catIds = coco.getCatIds()
    categories = coco.loadCats(catIds)
    category_names = [cat['name'] for cat in categories]

    # Подсчет общего количества изображений
    total_images = len(coco.getImgIds())
    print(f"Общее количество изображений {type}: {total_images}")

    mask_count = 0  # Счетчик сохраненных масок

    # Создание масок для каждого класса и изображения
    for cat_id, cat_name in zip(catIds, category_names):
        # Получение всех изображений для текущего класса
        imgIds = coco.getImgIds(catIds=[cat_id])
        imgDict = coco.loadImgs(imgIds)
        # Проходимся по каждой картинке определенного класса
        for im in imgDict:
            # Путь куда сохранять маску
            # file_path = os.path.join(output_dir + type, im['file_name'])
            filename_without_ext = os.path.splitext(im['file_name'])[0]
            file_path = os.path.join(output_dir + type, filename_without_ext + ".png")

            # Получаем айди аннотаций
            annIds = coco.getAnnIds(imgIds=[im['id']], catIds=[cat_id])
            anns = coco.loadAnns(annIds)

            # Было нужно в самом начале когда не совпадало количество фалйов с маской и без маски. Удалил не нужное, теперь все норм
            # if not anns:
            #     print(f"No annotations found for image ID {im['id']}")
            #     os.remove(im['file_name'])  # Удаление изображения без аннотаций
            #     continue

            # Инициализируем маску по размерам картинки
            mask = np.zeros((im['height'], im['width']), dtype=np.uint8)
            has_segmentation = False
            for ann in anns:
                if ann['segmentation']:  # Проверяем есть ли сегментация. Долго не мог понять почему многие дата сеты не работают
                    mask = np.maximum(mask, coco.annToMask(ann))
                    has_segmentation = True

            # Было нужно в самом начале когда не совпадало количество фалйов с маской и без маски. Удалил не нужное, теперь все норм
            if not has_segmentation:
                print(f"No valid segmentation found for image ID {im['id']}")
                os.remove(im['file_name'])  # Удаление изображения без сегментаций
                continue

            # Преоборазуем массив в изщображение в градации серого 0-255
            mask = Image.fromarray(mask * 255, mode="L")
            mask.save(file_path)
            mask_count += 1

    return mask_count


# Создание объектов COCO для train и val
coco_train = COCO(ANNOTATION_FILE_TRAIN)
coco_val = COCO(ANNOTATION_FILE_VAL)
coco_test = COCO(ANNOTATION_FILE_TEST)

# Генерация масок для train
mask_count_train = generate_masks(coco_train, rootPath, trainType)

# Генерация масок для val
mask_count_val = generate_masks(coco_val, rootPath, valType)

# Генерация масок для val
mask_count_test= generate_masks(coco_test, rootPath, testType)

print(f"Общее количество mask_train: {mask_count_train}")
print(f"Общее количество mask_valid: {mask_count_val}")
print(f"Общее количество mask_train: {mask_count_test}")
