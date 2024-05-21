import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Read image
img = cv2.imread('cats.jpg')
# Convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Plot it
plt.imshow(img)
# For ubuntu 22.04 in PyCharm need it to show
plt.show()


# Split the image into the three colour channels
red, green, blue = cv2.split(img)

# Compose the image in the RGB colour space
img1 = cv2.merge([red, green, blue])

# Compose the image in the RBG colour space
img2 = cv2.merge([red, green, blue])


# Compose the image in the GRB colour space
# img3 = cv2.merge([green, red, blue])
img3 = img1.copy()
img3[..., 0] = img1[..., 1]
img3[..., 1] = img1[..., 0]
img3[..., 2] = img1[..., 2]

# Compose the image in the BGR colour space
img4 = cv2.merge([blue, green, red])
# img4 =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Create the collage
out1 = np.hstack([img1, img2])
out2 = np.hstack([img3, img4])
out = np.vstack([out1, out2])

# Plot the collage
plt.imshow(out)
plt.axis(False)
plt.show()