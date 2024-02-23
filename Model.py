import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Directories containing images
data_dir1 = "/home/james/projects/circlesquareai-master/circle"
data_dir2 = "/home/james/projects/circlesquareai-master/square"

def load_images_from_dir(data_dir):
    images = []
    labels = []
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory.")
        return None, None
    for label, filename in enumerate(os.listdir(data_dir)):
        image_path = os.path.join(data_dir, filename)
        if not os.path.isfile(image_path):
            print(f"Error: {image_path} is not a file.")
            continue
        try:
            img = load_img(image_path, target_size=(255, 255))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image: {image_path}. {e}")
            continue
    return np.array(images), np.array(labels)

# Load images and labels from two directories
images1, labels1 = load_images_from_dir(data_dir1)
images2, labels2 = load_images_from_dir(data_dir2)

# Concatenate images and labels
images = np.concatenate([images1, images2])
labels = np.concatenate([labels1, labels2])

# Split the data into training, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(255, 255, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 classes: circle and square
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
