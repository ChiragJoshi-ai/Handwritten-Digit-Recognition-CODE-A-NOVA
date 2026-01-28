import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "trainingSet",
    image_size= (28, 28),
    color_mode= "grayscale",
    batch_size= 32,
    shuffle= True
)

print("Class Name:", train_ds.class_names)


def preprocess(x, y):
    x = x / 255.0                #divided by 255 because we are normalizing the pixle 
    x = 1 - x                    # 1 - x will invert that image
    return x, y

train_ds = train_ds.map(preprocess)


for images, labels in train_ds.take(2):
    plt.imshow(images[0].numpy().reshape(28, 28), cmap='gray')
    plt.title(f"label: {labels[0].numpy()}")
    plt.axis('off')
    plt.show()


#training a model

model = models.Sequential ([
    layers.Conv2D(32,(3,3), activation = 'relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_ds, epochs=10)
model.save("digit_model.h5")

#Predicting Image

def predict_image(img_path):
    print("Reading:", img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print("Image loaded:", img is not None)
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = 1 - img
    img = img.reshape(1,28,28,1)
    return np.argmax(model.predict(img))

valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
test_path = "testSet"
sample_imgs = [f for f in os.listdir(test_path) if f.lower().endswith(valid_ext)]

sample_imgs = sample_imgs[:5]       #only 5 images

for img_name in sample_imgs:
    img_path = os.path.join(test_path, img_name)
    pred = predict_image(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {pred}")
    plt.axis('off')
    plt.show()

