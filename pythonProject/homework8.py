# In this homework you are going to implement your first machine learning algorithm to automatically binarize document images. The goal of document binarization is to seprate the characters (letters) from everything else. This is the crucial part for automatic document understanding and information extraction from the . In order to do so, you will use the Otsu thresholding algorithm.

# At the end of this notebook, there are a couple of questions for you to answer.

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
# Let's load the document image we will be working on in this homework.

img = cv2.imread('data/document.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')
plt.show()
# First, let's have a look at the histogram.

h = np.histogram(img, 256)
plt.bar(h[1][0:-1], h[0])
plt.xlabel('Colour'), plt.ylabel('Count')
plt.grid(True)
plt.show()

# Let's now implement the Otsu thresholding algorithm. Remember that the algorithm consists of an optimization process that finds the thresholds that minimizes the intra-class variance or, equivalently, maximizes the inter-class variance.

# In this homework, you are going to demonstrate the working principle of the Otsu algorithm. Therefore, you won't have to worry about an efficient implementation, we are going to use the brute force approach here.
# Get image dimensions
rows, cols = img.shape
# Compute the total amount of image pixels
num_pixels = rows*cols

# Initializations
best_wcv = 1e6  # Best within-class variance (wcv)
opt_th = None  # Threshold corresponding to the best wcv

# Brute force search using all possible thresholds (levels of gray)
for th in range(0, 256):
    # Extract the image pixels corresponding to the background
    mask_f = img > th
    foreground = img[mask_f]
    # Extract the image pixels corresponding to the background
    mask_b = img <=th
    background = img[mask_b]

    # If foreground or background are empty, continue
    if len(foreground) == 0 or len(background) == 0:
        continue

    # Compute class-weights (omega parameters) for foreground and background
    # Пропорция пикселей
    omega_f = len(foreground)/num_pixels
    omega_b = len(background)/num_pixels
    # print(omega_f + omega_b) # Должно быть 1

    # Compute pixel variance for foreground and background
    # Hint: Check out the var function from numpy ;-)
    # https://numpy.org/doc/stable/reference/generated/numpy.var.html
    sigma2_f = np.var(foreground)
    sigma2_b = np.var(background)

    # Compute the within-class variance
    wcv = omega_f*sigma2_f + omega_b*sigma2_b

    # Perform the optimization
    if wcv < best_wcv:
        best_wcv = wcv # Подстава от Яна)))) Долго искал где ж я ошибся и чего ответы опенсиви импементации Отсу и это решение не совпадают. Оказало я не внимателен и просто присваивал нигде не испорльзующейся переменной  best значения. Получалось что следующая итерация проверки шла лесом)))
        opt_th = th

# Print out the optimal threshold found by Otsu algorithm
print('Optimal threshold', opt_th)

opt_th_by_opencv_implement, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
print("opt_th_by_opencv_implement = ", opt_th_by_opencv_implement)


# Finally, let's compare the original image and its thresholded representation.

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(img > opt_th, cmap='gray')
plt.show()

# Questions
# Looking at the computed histogram, could it be considered bimodal?
# Пришлось гуглить что значит бимодальная гистограмма)) И так, если она имеет 2 отчетливых пика, то ее можно считать бимодальной.
# Конкретно в нашем случае, похоже что мы имеем 3 пика(белый, ярковыраженный светло серый и слабо выраженй серый). Значит поидеи у нас мультимодальная гистограмма


# Looking at the computed histogram, what binarization threshold would you chose? Why?
# Я бы наверное выбрал трешхолд примерно 220. Это из-за того что визуально вижу два ярко выряженных пика в гистограмме и третий не большой(серый), которй мог бы посчитать не пиком из-за визалього восприятия.
# Если честно, из-за опечатки от Яна с best_wcv, я думал что вначале нашел правильные значения трешхолда. Я не придал значение, что значение 254, думал что так значит правильно.
# но потом когда вывел имплементацию опесиви, то понял шо где-то ошибся


# Looking at the resulting (thresholded) image, is the text binarization (detection) good?
# По сути вижу эффект сканирования копий документа или множественной печати на принтере. Вообще результат не плохой. Хотя видно, что некотрые слова теперь нельзя разобрать


