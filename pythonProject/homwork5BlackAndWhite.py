import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Load image
img = cv2.imread('data/kodim23.png')
# Convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Plot it
plt.imshow(img)
plt.show()

# Let's start with gray tones first.

# Black, dark gray, light gray, white
colors = np.array([[0, 0, 0],
                   [255, 255, 255]])
# Using the colour pallette, let's quantize the original image

# Cast the image to float
img = img.astype(np.float32)

# Prepare for quantization
rows, cols, channels = img.shape
quantized = np.zeros_like(img)

# Apply quantization
for r in range(rows):
    for c in range(cols):
        # Extract the original pixel value
        old_pixel = img[r,c, :]

        # Find the closest colour from the pallette (using absolute value/Euclidean distance)
        # Note: You may need more than one line of code here
        # Euclidean distance https://en.wikipedia.org/wiki/Euclidean_distance
        # Судя по всему в этом месте нужен axis=1. Без него получаю черную картинку. Как я понял этот параметр указывает как будет идти вычисления. Если убрать его, то будет считать одно общее значения для массива, а если поствить, то будет считать по строчно.
        # euclideanDistance = np.sqrt(np.sum((colors - old_pixel) ** 2))
        euclideanDistance = np.sqrt(np.sum((colors - old_pixel) ** 2, axis=1))
        # print(euclideanDistance)
        new_pixel = colors[np.argmin(euclideanDistance)]  # Находим ближайшее значение из масива colors к нашей дистанции

        # Apply quantization
        quantized[r, c, :] = new_pixel

# Show quantized image (don't forget to cast back to uint8)
plt.imshow(quantized.astype(np.uint8))
plt.show()

def psnr(ref, target):
    error = ref.astype(np.float32) - target.astype(np.float32)
    mse = np.mean(error**2)
    return 10 * np.log10((255**2)/mse)

# Compute average quantization error
# avg_quant_error =
PSNR = psnr(img, quantized)
print(PSNR)


# Floyd-Stenberg Dithering
# Make a temporal copy of the original image, we will need it for error diffusion
pixels = np.copy(img)
dithering = np.zeros_like(img)

for r in range(1, rows - 1):
    for c in range(1, cols - 1):
        # Extract the original pixel value
        old_pixel = pixels[r, c, :]
        # Find the closest colour from the pallette (using absolute value/Euclidean distance)
        # Note: You may need more than one line of code here
        euclideanDistance = np.sqrt(np.sum((colors - old_pixel) ** 2, axis=1))
        # print(euclideanDistance)
        new_pixel = colors[np.argmin(euclideanDistance)]  # Находим ближайшее значение из масива colors к нашей дистанции

        # Тут есть отличный пример https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
        # Compute quantization error
        quant_error = old_pixel - new_pixel

        # Diffuse the quantization error accroding to the FS diffusion matrix
        # Note: You may need more than one line of code here
        pixels[r, c] = new_pixel
        pixels[r, c + 1] += quant_error * 7 / 16
        pixels[r + 1, c - 1] += quant_error * 3 / 16
        pixels[r + 1, c] += quant_error * 5 / 16
        pixels[r + 1, c + 1] += quant_error * 1 / 16

        # Apply dithering
        dithering[r, c, :] = new_pixel

# plt.imshow(img_tmp)
# plt.show(dithering)
# Show quantized image (don't forget to cast back to uint8)
plt.subplot(121), plt.imshow(quantized.astype(np.uint8)), plt.title('Optimally quantized')  # optimally quantized
plt.subplot(122), plt.imshow(dithering.astype(np.uint8)), plt.title('dithering') # dithering
plt.show()

# Compute average quantization error for dithered image
PSNR = psnr(img, dithering)
print(PSNR)
