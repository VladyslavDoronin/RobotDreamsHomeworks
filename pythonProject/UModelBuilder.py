import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from CustomDataGenerator import CustomDataGenerator
from tensorflow.keras.preprocessing.image import load_img

plt.rcParams['figure.figsize'] = [15, 10]
trainType = "train"
valType = "valid"
testType = "test"

ANNOTATION_FILE_TRAIN = f'/home/user/Documents/data/MilVehicle/{trainType}/_annotations.coco.json'
ANNOTATION_FILE_VAL = f'/home/user/Documents/data/MilVehicle/{valType}/_annotations.coco.json'
ANNOTATION_FILE_TEST = f'/home/user/Documents/data/MilVehicle/{testType}/_annotations.coco.json'

root = "/home/user/Documents/data"
input_dir_train = os.path.join(root, "MilVehicle/train")
target_dir_train = os.path.join(root, "Masks/mask_train")

input_dir_valid = os.path.join(root, "MilVehicle/valid")
target_dir_valid = os.path.join(root, "Masks/mask_valid")

input_dir_test = os.path.join(root, "MilVehicle/test")
target_dir_test = os.path.join(root, "Masks/mask_test")

class_names = ['btr', 'tank', 'truck']
class_ids = [1, 2, 3]


batch_size = 16

# Построение модели сегментации
def build_unet_model(input_shape):
    inputs = Input(shape=input_shape)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    # c1 = layers.BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    # c2 = layers.BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    # c3 = layers.BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    # c4 = layers.BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)


    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)


    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid', padding="same")(c9)
    # outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    # outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)


    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# input_shape = (640, 480, 3)
input_shape = (128, 128, 3)
model = build_unet_model(input_shape)
model.summary()
plot_model(model, show_shapes=True)
plt.show()

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [keras.callbacks.ModelCheckpoint("unetSegmentation.keras", save_best_only=True)]

# Создание генераторов данных для обучения и валидации
# train_generator = CustomDataGenerator(input_dir_train, target_dir_train, 8)
train_generator_tank = CustomDataGenerator(images_path=input_dir_train, masks_path=target_dir_train, img_size=(128, 128),
                                          annotation_file=ANNOTATION_FILE_TRAIN, class_id=class_ids[1],
                                          batch_size=batch_size, shuffle=True)
# train_generator_tank = CustomDataGenerator(images_path=input_dir_train, masks_path=target_dir_train, annotation_file=ANNOTATION_FILE_TRAIN, class_id=class_ids[1], batch_size=batch_size)

valid_generator_tank = CustomDataGenerator(images_path=input_dir_valid, masks_path=target_dir_valid,
                                           img_size=(128, 128),
                                           annotation_file=ANNOTATION_FILE_VAL, class_id=class_ids[1],
                                           batch_size=batch_size, shuffle=True)
# valid_generator_tank = CustomDataGenerator(images_path=input_dir_valid, masks_path=target_dir_valid, annotation_file=ANNOTATION_FILE_VAL, class_id=class_ids[1], batch_size=batch_size)
# Загрузка данных для каждого класса
# images, tank_masks = load_data(image_folder, annotation_file, class_ids[0])
# _, btr_masks = load_data(image_folder, annotation_file, class_ids[1])
# _, bus_masks = load_data(image_folder, annotation_file, class_ids[2])

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_generator_tank, epochs=epochs, validation_data=valid_generator_tank, callbacks=callbacks)

def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-', epochs, h['val_accuracy'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    print('Train Acc     ', h['accuracy'][-1])
    print('Validation Acc', h['val_accuracy'][-1])


plot_history(model)

test_generator_tank = CustomDataGenerator(images_path=input_dir_test, masks_path=target_dir_test, img_size=(128, 128),
                                          annotation_file=ANNOTATION_FILE_TEST, class_id=class_ids[1],
                                          batch_size=batch_size, shuffle=True)
# test_generator_tank = CustomDataGenerator(images_path=input_dir_test, masks_path=target_dir_test, annotation_file=ANNOTATION_FILE_TRAIN, class_id=class_ids[1], batch_size=batch_size)
test_preds = model.predict(test_generator_tank)

idx = 2

mask = np.argmax(test_preds[idx], axis=-1)
mask = np.expand_dims(mask, axis=-1)

print(test_generator_tank.image_files[idx])
print(test_generator_tank.mask_files[idx])
img = cv2.cvtColor(cv2.imread(test_generator_tank.image_files[idx]), cv2.COLOR_BGR2RGB)
maskTrue = cv2.imread(test_generator_tank.mask_files[idx])

rows, cols, _ = maskTrue.shape
mask = cv2.resize(mask.astype(np.uint8), (cols, rows))

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(maskTrue/255)
plt.subplot(133), plt.imshow(mask, cmap='gray')
plt.show()

#
# x = load_img(test_generator_tank.image_files[idx])
# y_true = load_img(test_generator_tank.mask_files[idx])
# y_true = (np.array(y_true) - 1.0)/2
#
# rows, cols, _ = y_true.shape
# mask = cv2.resize(mask.astype(np.uint8), (cols, rows))
#
# plt.subplot(131), plt.imshow(x)
# plt.subplot(132), plt.imshow(y_true)
# plt.subplot(133), plt.imshow(mask, cmap='gray')
# plt.show()

model.save('data/final_model.h5')

