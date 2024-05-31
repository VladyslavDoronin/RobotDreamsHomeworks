import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('data/kodim01.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Самый ектримальный случай размытия с размером 3х3
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9  # Фильтр с такой структурой
rows, cols, channels = img.shape
out = np.zeros_like(img)  # Заполняем масив нулями точно такой же как у картинки

# Sliding window applied to each colour channel
# Это обычнй фильтр(свертка или конволюция), где происходит сгаживание.
for r in range(1, rows - 1):
    for c in range(1, cols - 1):
        for ch in range(0, channels):
            block = img[r - 1:r + 2, c - 1:c + 2, ch]
            out[r, c, ch] = np.sum(block * kernel)

plt.imshow(out)
plt.show()


# Еффект четкости(sharpering)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# cv2.filter2D это тоже самое что написано выше в цикле фор, только на много быстрее
#  ddepth=-1 означает что я не хочу ничего менять и тип картинки какой был, тако и останется
# orderType=cv2.BORDER_REPLICATE означает что я не хочу менять размер картинки, чтоб она не обрезалась при наложении фильтров на крайние значения пикселей. Используется самый близкий цвет по координатам
out = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
plt.imshow(out)
plt.show()
print(kernel)

# Or even simpler for predefined filters
out = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=5)
plt.imshow(out)
plt.show()


out = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
plt.imshow(out)
plt.show()