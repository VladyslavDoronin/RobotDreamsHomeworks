# HOMEWORK 13
# In this homework you are going to inspect the GTSDB (German Traffic Sign Detection Benchmark) dataset. The dataset contains images of various classes of traffic signs used in Germany (and the whole EU). The objective of this homework is to go through the steps described below and to implement the necessary code.
#
# At the end, as usual, there will be a couple of questions for you to answer. In addition, the last section of this homework is optional and, if you chose to do it, you'll earn extra point :-)

import os
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Step 0
# Go to the GTSRB dataset official site (link) to learn more about the dataset.
# И так из интересного:
# Более 40 классов
# Более 50к фото
# Размеры картинки варируются от 15х15 до 250х250
# Не все картинки квадратной формы
# Не все картинки по середине

# Step 1
# Download the dataset (link) and unzip it.
#
# Step 2
# For this homework, you will be working with the training set. Check out the Train.csv, open it and see what it contains. Load the dataset and plot random samples.
# Load the training labels
root = 'data/GTSRB' # Path to the dataset location, e.g., '/data/janko/dataset/GTSRB'
data = pd.read_csv(os.path.join(root, 'Train.csv'))

# Number of training samples (amount of samples in data)
num_samples = len(data)

# Show random data samples
for ii in range(15):
    # Get random index
    idx = np.random.randint(0, num_samples)
    # Load image
    img = cv2.imread(os.path.join(root, data.iloc[idx]['Path']))
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Show image
    plt.subplot(3,5,ii+1), plt.imshow(img), plt.title(data.iloc[idx]['ClassId'])

plt.show()
#Вижу что дейстивтельно разные размеры картинок и при этом бывают еще и прямоугольные и сжатые по вертикали или горизонтали

# Step 3
# Inspect the dataset by computing and plotting the per-class histogram.

# Extract class identifiers
# Hint: Check the csv
ids = data['ClassId']

# Compute the per class histogram. You can use any approach you want (e.g. numpy). It's also worth looking at the Counter function from the collections module (link) ;-)

from collections import Counter
hist = Counter(ids)

plt.bar(hist.keys(), hist.values()), plt.grid(True)
plt.xlabel('ID'), plt.ylabel('Counts')
plt.show()

# Questions
# Please answer the following questions:
#
# Do you consider the dataset to be balanced? If so, why? If not, why?
# Датасет очень сильно НЕ сбалансирован.
# Тут количество объектов разных классов могут быть от 200 до 2000+. Это считай в 10 раз разница
# Для хорошей балансировки количество фоток каждого класса должно быть примерно одинаковое



# Are there any classes that are (significantly) over-represented or under-represeneted?
# На самом деле из предыдущего вопроса и ответа можно уже сделать вывод, что тут есть и over-represented и under-represeneted классы.
# Но давайте еще для наглядности выведем относительное выражение гстаграммы
print(hist.items())
total = hist.total()
print(total)
normalized_hist = {key: value / total for key, value in hist.items()}

plt.bar(normalized_hist.keys(), normalized_hist.values()), plt.grid(True)
plt.show()
print(100/42)
# И так, тут есть и черезмерно представленные классы и недопредставленные. некоторые классы представленны до 1%, в то время другие - более 6%
# Для хорошего распределения тут должно быть представленно 100%/42класса=2.38% каждого класса
# Соответственно черезмерно представленные классы тут те которые занимают более 6%(я б взял даже от 5%)
# И недостаточно предлставленны те классы которые имеют менее 1%


# Optional
# Perform a further analysis on the dataset and draw some conclusion from it.
#
# Hint 1: Unlike MNIST or CIFAR10, this dataset contains images with various spatial resolutions. Is there anything we can tell about the resolution distribution? Hint 2: What about the brightness distribution? Are there classes there are significantly more bright than others?
# Уже визуально видно что картинки имеют разные разрешения. Та и на сайте где описан дата сет так сказано. Но давайте выведем эти размеры

resolutions = [cv2.imread(os.path.join(root, path)).shape[:2] for path in data['Path']]
print(resolutions)
# Через print видно что размеры тут самые разные и оджинаковых практически не видно сразу невооруженным глазом. Выводит в гистограмму нет смысла



# Давайте проверим теперь по яркости классы
brightness = [cv2.imread(os.path.join(root, path), cv2.IMREAD_GRAYSCALE).mean() for path in data['Path']]

data['Brightness'] = brightness

data.groupby('ClassId')['Brightness'].mean().plot(kind='bar')
plt.xlabel('ID')
plt.ylabel('Mean Brightness')
plt.title('Mean Brightness per Class')
plt.show()
# По яркости классы тоже довольно сильно отличаются