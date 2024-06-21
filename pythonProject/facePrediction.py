# Step 0
# Run the necessary imports.
import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

# Step 1
# Load an image (any image that contains faces).
images = [
    ('data/my.jpg', 'My img'),
    ('data/my2.jpg', 'My2 img'),
    ('data/many.jpg', 'Many Faces'),
    ('data/many2.jpg', 'Many Faces 2'),
    ('data/blackManDark.jpg', 'Black Man Dark'),
    ('data/BlackManLight.webp', 'Black Man  Light'),
    ('data/closeEyes.jpeg', 'Closed Eyes'),
    ('data/handsOnEyes.jpg', 'Hands on Eyes'),
    ('data/mask.jpeg', 'Mask'),
    ('data/noEyes.jpg', 'No Eyes'),
    ('data/noFace.jpg', 'No Face'),
    ('data/skeleton.jpeg', 'Skeleton')
]

fig, axes = plt.subplots(4, 3, figsize=(15, 15))
axes = axes.flatten()

for ax, (image_path, title) in zip(axes, images):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ax.imshow(img), ax.set_title(title)
        ax.axis('off')
    else:
        ax.set_title(f'Failed to load {title}')
        ax.axis('off')

fig.suptitle('Origin', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Step 2
# Load the dlib face predictor or others.
# Step 3
# Run the predictor on your image.
# Step 4
# Draw bounding boxes around the detected faces and plot the image. Use different colour for each face.
# Step 5 (optional)
# Repeat the process with a different and more challenging image (more faces, smaller faces, people with glasses, hats, helmets, etc.). How does the detector perform? Is it robust?

#_____________________________DLIB_______________________________________
detector = dlib.get_frontal_face_detector()

fig, axes = plt.subplots(4, 3, figsize=(15, 15))
axes = axes.flatten()

for ax, (image_path, title) in zip(axes, images):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)

        print('Number of detected faces:', len(rects))
        result_dlib = np.copy(img)
        if len(rects) > 0:
            # print(rects)
            # print(rects[0].left)
            faces_dlib_img = []
            for rect in rects:
                x, y, w, h = rect_to_bb(rect)
                print(x, y, w, h)
                cv2.rectangle(result_dlib, (x, y), (x + w, y + h), (255, 255, 0), 3)
                faces_dlib_img.append(img[y:y + h, x:x + w, :])
        ax.imshow(result_dlib), ax.set_title(title)
        ax.axis('off')
    else:
        ax.set_title(f'Failed to load {title}')
        ax.axis('off')

fig.suptitle('Dlib', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
# plt.show()


#_____________________________Voila-Jones______________________________________-
casc_path = 'model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(casc_path)

fig, axes = plt.subplots(4, 3, figsize=(15, 15))
axes = axes.flatten()

for ax, (image_path, title) in zip(axes, images):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

        print('Number of detected faces:', len(faces))

        # Draw rectangle around each face
        result = np.copy(img)
        faces_img = []
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            faces_img.append(img[y:y + h, x:x + w, :])
        ax.imshow(result), ax.set_title(title)
        ax.axis('off')
    else:
        ax.set_title(f'Failed to load {title}')
        ax.axis('off')

fig.suptitle('Voila-Jones', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# How does the detector perform? Is it robust?
#  И так, я использовал разные картинки, с лицами и без, черная кожа и светлая, зарытое лицо и в масках и тд.
#  Из полученных результатов могу сделать вывод, что Voila-Jones опередляет лица немного лучше dlib.
#  Хотя тут зависсит от параметра minNeighbors. Чем меньше этот параметр выставить, тем больше ложных целей(лица у деревьев) он находит. И чем больше этоот параметр тем меньше лиц определяет реальных
# По сути тут хорошие результаты по детекции лиц, но не идельно - много равльного не найдено.
# Лишние предметы на лице типо маски или очков мешают определить что это лицо. Так же, если лицо под наклоном, то можент его не определить как лицо. Или даже повернуто лицо немного вбок, тоже не определяет нормально



