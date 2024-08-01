import os
import matplotlib.pyplot as plt
from random import shuffle
from pycocotools.coco import COCO

# Пути к датасету COCO
ANNOTATION_FILE_TRAIN = '../../../data/MilVehicle/train/_annotations.coco.json'
ANNOTATION_FILE_VAL = '../../../data/MilVehicle/valid/_annotations.coco.json'

root = "data"
input_dir = os.path.join(root, "MilVehicle/train")
target_dir = os.path.join(root, "Masks/mask_train")

input_img_paths = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
input_img_paths = sorted([os.path.join(input_dir, fname) for fname in input_img_paths])

target_img_paths = [f for f in os.listdir(target_dir) if f.endswith(".png") and not f.startswith("_")]
target_img_paths = sorted([os.path.join(target_dir, f) for f in target_img_paths])

print(len(input_img_paths))
print(len(target_img_paths))
assert len(input_img_paths) == len(target_img_paths)
print("Number of samples:", len(input_img_paths))

# Создание объекта COCO train
coco_train = COCO(ANNOTATION_FILE_TRAIN)
# Получение всех категорий
categories_train = coco_train.loadCats(coco_train.getCatIds())
category_names_train = [cat['name'] for cat in categories_train]
imgIds_train = [cat['id'] for cat in categories_train]
imgDict_train = coco_train.loadImgs(imgIds_train)
print(imgDict_train)

# Создание объекта COCO validation
coco_val = COCO(ANNOTATION_FILE_VAL)
# Получение всех категорий
categories_val = coco_val.loadCats(coco_val.getCatIds())
category_names_val = [cat['name'] for cat in categories_val]
imgIds_val = [cat['id'] for cat in categories_val]
imgDict_val = coco_train.loadImgs(imgIds_val)

# Выводим количество train и validation картинок и классов
print("Train ids len: ", len(imgIds_train), "Train class names len:", len(category_names_train))
print("Val ids len: ", len(imgIds_val), "Val class names len:", len(category_names_val))

# Перемешиваем датасет
shuffle(imgIds_train)
shuffle(imgIds_val)

# Инициализация словаря для подсчета количества аннотаций по каждой категории
category_annotation_counts = {cat_id: 0 for cat_id in imgIds_train}

# Получение всех ID аннотаций
annIds = coco_train.getAnnIds()
annotations = coco_train.loadAnns(annIds)

# Подсчет количества аннотаций по каждой категории
for ann in annotations:
    category_id = ann['category_id']
    category_annotation_counts[category_id] += 1

# Создание списка количества аннотаций по каждой категории
annotation_counts = [category_annotation_counts[cat_id] for cat_id in imgIds_train]

# Визуализация распределения аннотаций
plt.figure(figsize=(20, 20))
plt.bar(category_names_train, annotation_counts)
plt.xlabel('Классы')
plt.ylabel('Количество')
plt.title('Отношение количества аннотаций к классам')
plt.show()

# Печать информации о балансировке
total_annotations = sum(annotation_counts)
for cat_id, cat_name in zip(imgIds_train, category_names_train):
    count = category_annotation_counts[cat_id]
    percentage = (count / total_annotations) * 100
    print(f'Класс: {cat_name}, Количество: {count}, Соотношение: {percentage:.2f}%')
