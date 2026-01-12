import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image, resize it, and normalize it.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

def create_dataset(data_dir, target_size=(224, 224)):
    """
    Create a dataset of images and labels from a directory.
    Assumes that the data_dir contains subdirectories for each class.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = load_and_preprocess_image(image_path, target_size)
            if img is not None:
                images.append(img)
                labels.append(i)
    return np.array(images), np.array(labels)

def get_data_generators(X, y, n_splits=5, batch_size=32):
    """
    Create data generators for cross-validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

        val_datagen = ImageDataGenerator()
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

        yield train_generator, val_generator
