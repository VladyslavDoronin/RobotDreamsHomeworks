# Face Detection
# Face detection is a crucial step for many applications, especially for personal identification and access control. In this notebook, we are going to demonstrate the use of several tool, namely:
#
# Viola-Jones object detector (applied to faces)
# concept of facial landmarks
# dlib library
# face alignment
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
# We are going to load the Viola-Jones detector based on Haar cascades. The detector, already trained for detecting faces, is part of opencv contrib distribution and is located in its data subfolder.

casc_path = '/home/user/Documents/GitHub/RobotDreamsHomeworks/pythonProject/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(casc_path)
img = cv2.imread('data/myPhoto.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img)
plt.show()

# Viola - Jones
# Viola - Jones is a
# classic and powerful
# algorithm
# for object detection.It is a sliding window approach that work in cascades and exploits Haar transform (basis functions) to learn object descriptors.It also makes uso of boosting.

# minNeighbors = 0 shows all the detection at all scale, a value of approx. 5 shall felter out all the spurious detections
# Испольуем разные масштабы для детекции лица
# minNeighbors делает проверку есть ли подтверждение что найдено лицо еще вокруг найденного набора(пикселей подтвержденных что это лицо). Чем мень тем мень проверок вокруг будет и тем вероятнее что нйдет лица даже у деревьев
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, flags=cv2.CASCADE_SCALE_IMAGE)

print('Number of detected faces:', len(faces))

# Draw rectangle around each face
result = np.copy(img)
faces_img = []
for (x, y, w, h) in faces:
    # Draw rectangle around the face
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
    faces_img.append(img[y:y + h, x:x + w, :])

plt.imshow(result)
plt.show()
plt.subplot(121), plt.imshow(result, cmap='gray')
plt.subplot(122), plt.imshow(faces_img[0])
plt.show()


# Face Alignment
# Face alignment is an important pre-processing step for face identification. The goal is to transform the detected faces as it was taken in frontal view. In general, only affine transforms are considered so the facial proportions are maintained.

face_template = np.load('data/face_template.npy')

tpl_min, tpl_max = np.min(face_template, axis=0), np.max(face_template, axis=0)
face_template = (face_template - tpl_min) / (tpl_max - tpl_min)

plt.plot(face_template[:, 0], -face_template[:, 1], 'o')

inner_triangle = [39, 42, 57]
plt.plot(face_template[inner_triangle, 0], -face_template[inner_triangle, 1], 'rs')

plt.axis('square')
out_size = 256
margin = 10
out = np.copy(face)

# Prepare landmarks
landmarks = np.float32(landmarks)
landmarks_idx = np.array(inner_triangle)

# Adjust template (adjust to size, to margin and normalize back)
template = face_template * out_size
template = template + (margin/2)
template = template / (out_size + margin)

# Estimate affine transform
H = cv2.getAffineTransform(landmarks[landmarks_idx], (out_size + margin) * template[landmarks_idx])

# Rectify final image
aligned = cv2.warpAffine(out, H, (out_size + margin, out_size + margin))

plt.subplot(121), plt.imshow(face), plt.title('Detected')
plt.subplot(122), plt.imshow(aligned), plt.title('Aligned')