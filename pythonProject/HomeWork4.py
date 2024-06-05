# For this homework you are going to implement a lane line detector. Lane line detection is crucial for ADAS (Advanced Driver Assistance Systems) systems and, in particular, for LKA (Lane Keep Assist). You will use a picture from a front facing camera (mounted on the car) and will implement the following steps:

# Convert image to gray scale
# Compute edge map
# Apply Hough transform to obtain line parametrizations
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
# Let's load and show the camera frame.

img = cv2.imread('data/dashcam.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
plt.imshow(img)
plt.show()
# Convert image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Obtain edge map
# Hint: you can use Canny edge detector with th_low = 100, th_high = 150
# edges = cv2.Canny(gray, threshold1=50, threshold2=100)  # Много шума
# edges = cv2.Canny(gray, threshold1=50, threshold2=300)  # Обрезало много, пол машины нет
# edges = cv2.Canny(gray, threshold1=150, threshold2=300)  # Тоже слишком много убрало
# edges = cv2.Canny(gray, threshold1=100, threshold2=150)  # Тут видны дворники машины
edges = cv2.Canny(gray, threshold1=120, threshold2=190)  # Как по мне эти пороги лучше подходят чем в подсказке. Визуально это хоть дворники машины убирает

plt.imshow(edges, cmap='gray')
plt.show()

# We are only interseted in the road so we will remove everything above the horizon
print("edges: ", edges.shape)
edges[0:350] = 0
# plt.imshow(edges, cmap='gray')
# Let's plot the images
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge map')
plt.show()

# Apply Hough transform to parametrize the lines
# Hint 1: Offset resolution of 2 pixels and slope resolution of 2 degrees work well in this case
# Hint 2: A suitable value for the accumulator threshold is 190
print("theta: ", 2 * np.pi / 180)
lines = cv2.HoughLines(edges, rho=2, theta=2 * np.pi / 180, threshold=190)
# Let's get rid of the unnecessary dimension
print("lines: ", lines.shape)
lines = lines[:, 0, :]

# Plot the resulting Hough lines
result = np.copy(img)

for line in lines:
    # print("line: ", line.shape)
    rho = line[0]
    theta = line[1]
    print("rho: ", rho)
    print("theta: ", theta)

    a = math.cos(theta)
    b = math.sin(theta)

    x0 = a * rho
    y0 = b * rho

    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    cv2.line(result, pt1, pt2, 255, 1, cv2.LINE_AA)

plt.subplot(121), plt.imshow(edges, cmap='gray'), plt.title('Edge map')
plt.subplot(122), plt.imshow(result, cmap='gray'), plt.title('Hough lines')
plt.show()

# The edge map looks good but the Hough lines are too noisy. Let's clean the Hough lines first by removing all lines that we know cannot represent a lane line. In other words, all lines that are approximately horizontal shall be removed. Remember that horizontal lines correspond to theta = 90 degrees.

# Filter out all lines that are approximately horizontal (+/- 20 degrees).
filtered_lines = []
for line in lines:
    # Extract theta for current line (remember Hough works with radians)
    theta = line[1]
    print("filtered_lines theta: ", theta)
    # Keep line if theta is not horizontal
    theta_Rad_Minus_20 = (90-20)*np.pi/180
    theta_Rad_Plus_20 = (90+20)*np.pi/180
    minus_theta_Rad_Minus_20 = (-90-20)*np.pi/180
    minus_theta_Rad_Plus_20 = (-90+20)*np.pi/180
    print("theta_Rad_Minus_20: ", theta_Rad_Minus_20)
    print("theta_Rad_Plus_20: ", theta_Rad_Plus_20)
    print("minus_theta_Rad_Minus_20: ", minus_theta_Rad_Minus_20)
    print("minus_theta_Rad_Plus_20: ", minus_theta_Rad_Plus_20)

    if not (theta_Rad_Minus_20 < theta < theta_Rad_Plus_20 or
            minus_theta_Rad_Minus_20 < theta < minus_theta_Rad_Plus_20):
        filtered_lines.append(line)

# Let's plot the resulting filtered lines
result = np.copy(img)

for line in filtered_lines:
    # print("line: ", line.shape)
    rho = line[0]
    theta = line[1]
    # print("rho: ", rho)
    # print("theta: ", theta)

    a = math.cos(theta)
    b = math.sin(theta)

    x0 = a * rho
    y0 = b * rho

    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    cv2.line(result, pt1, pt2, 255, 1, cv2.LINE_AA)

plt.subplot(121), plt.imshow(edges, cmap='gray'), plt.title('Edge map')
plt.subplot(122), plt.imshow(result, cmap='gray'), plt.title('Hough lines')
plt.show()

# The result is now much better, but still we see some very similar lines. How can we get rid of them?

# Let's apply k-means clustering. It will find the clusters of the 6 we see in the picture lines and use the averages.
# We will apply k-means clustering to refine the detected lines.
# Don't worry, we will learn about the clustering later in the course :-)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6).fit(filtered_lines)
print("kmeans ", kmeans)
centers = kmeans.cluster_centers_
print("center: ", centers)

# Again, let's plot the resulting filtered lines
result = np.copy(img)

for line in kmeans.cluster_centers_:
    # print("line: ", line.shape)
    rho = line[0]
    theta = line[1]
    # print("rho: ", rho)
    # print("theta: ", theta)

    a = math.cos(theta)
    b = math.sin(theta)

    x0 = a * rho
    y0 = b * rho

    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    cv2.line(result, pt1, pt2, 255, 1, cv2.LINE_AA)

plt.subplot(121), plt.imshow(edges, cmap='gray'), plt.title('Edge map')
plt.subplot(122), plt.imshow(result, cmap='gray'), plt.title('Hough lines')
plt.show()

# Questions

# Do you see anything strange in the final result?
# Да, вижу что линия которую рисуем не совсем прилигает(совпадает) к линии дороги.
# По предыдущему результату, где видно все линии и не убрали еще лишнии, видно, что есть линии прилегающие к начальной части дороги, но потом отлегают
# И потом есть линии на дальней части дороги, которые в самом начале не совпадают с линией.
# Судя по всему есть какой-то исгиб линий на самом изображении, что они не совсем прямые или это какой-то эффект исгибания линий при отдалении линий, который просто не знаю как нащзывается.
# Или еще предположение, что может быть какие-то ошибки или не совем правильное вычисление границ.
# Или не совем точное вычисление по Хафу. В обоих случая может играть роль округления значений.
# Судя по второму вопросу действительно дело в вычислении по Хафу


# Do you think the Hough transform resolution is important for obtaining a good result? Why?
# Да, преобразование Хафа важно для лучшего результата. Преобразование Хафа помогает лучше определить линии

# Do you think the Hough transform accumulator threshold is important for obtaining a good result? Why?
# Да, threshold Хафа важен. Он определяет сколько пикселей пересекает линия.
# Чем больше ставить threshold, тем более ярко выраженную линию будет показывать и может пропустить правильную линию.
# А чем меньше threshold, тем больше шума оно тоже может оперделять как линии