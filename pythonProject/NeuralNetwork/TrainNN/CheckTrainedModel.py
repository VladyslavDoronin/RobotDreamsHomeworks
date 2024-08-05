import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Определяем путь к картинке
new_image_path = '/home/user/Downloads/photo_2024-08-02_20-02-52.jpg'

# грузим картинку и изменяем ее размеры под размеры которым обучалась нейронка
new_image = load_img(new_image_path, target_size=(128, 128))
new_image_array = img_to_array(new_image) / 255.0  # Нормализуем картинку

# Добавляем batch. Нейронка была обучена с ним и model.predict ожидает batch
new_image_array = np.expand_dims(new_image_array, axis=0)

# Загружаем нашу обученную модель
# Файлик не коммичу на гитхаб Силшком много весит. Этот файлик можно получить тут
# https://drive.google.com/file/d/1X5lq5kUzBdhq_ntEud91zUFLd_w8qWLV/view?usp=sharing
model = load_model('NeuralNetwork/TrainResults/unetSegmentation.keras')

# Пытаемся найти объект для сегментации, Делаем предсказание
new_image_pred = model.predict(new_image_array)

# Обрабатываем предсказание дла наложения маски
new_image_mask = np.argmax(new_image_pred[0], axis=-1)
new_image_mask = np.expand_dims(new_image_mask, axis=-1)

# Грузим оригинальную картинку для отображения радом с маской, для визуализации результата
original_img = cv2.cvtColor(cv2.imread(new_image_path), cv2.COLOR_BGR2RGB)

# изменяем размеры маски по истинные размеры картинки
rows, cols, _ = original_img.shape
new_image_mask_resized = cv2.resize(new_image_mask.astype(np.uint8), (cols, rows))

# ВЫводим результат
plt.subplot(121), plt.imshow(original_img)
plt.subplot(122), plt.imshow(new_image_mask_resized, cmap='gray')
plt.show()

