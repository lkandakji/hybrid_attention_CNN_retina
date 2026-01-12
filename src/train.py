import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import create_dataset, get_data_generators
from model import create_ha_cnn_model

# --- Configuration ---
DATA_DIR = 'data/'  # Directory containing the dataset
TARGET_SIZE = (224, 224)
NUM_CLASSES = 4  # Normal, DR, AMD, Glaucoma
N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 1

def train_and_evaluate():
    """
    Main function to train and evaluate the model.
    """
    # 1. Load and preprocess the data
    print("Loading and preprocessing data...")
    X, y = create_dataset(DATA_DIR, target_size=TARGET_SIZE)
    print(f"Loaded {len(X)} images.")

    # 2. Perform cross-validation
    fold = 1
    all_preds = []
    all_true = []

    for train_generator, val_generator in get_data_generators(X, y, n_splits=N_SPLITS, batch_size=BATCH_SIZE):
        print(f"--- Training Fold {fold}/{N_SPLITS} ---")

        # 3. Create the model
        model = create_ha_cnn_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=NUM_CLASSES)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 4. Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=len(val_generator)
        )

        # 5. Evaluate the model on the validation set
        X_val, y_val = val_generator.x, val_generator.y
        y_pred = np.argmax(model.predict(X_val), axis=1)
        all_preds.extend(y_pred)
        all_true.extend(y_val)

        fold += 1

    # 6. Generate and save classification report and confusion matrix
    print("\n--- Overall Performance ---")
    class_names = sorted(os.listdir(DATA_DIR))
    report = classification_report(all_true, all_preds, target_names=class_names, zero_division=0)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    # Set up a dummy dataset for testing purposes
    class_names = ['normal', 'dr', 'amd', 'glaucoma']
    should_create_dummy_data = False
    for class_name in class_names:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir) or not os.listdir(class_dir):
            should_create_dummy_data = True
            break

    if should_create_dummy_data:
        print("Creating a dummy dataset for testing...")
        for class_name in class_names:
            os.makedirs(os.path.join(DATA_DIR, class_name), exist_ok=True)
            for i in range(20):
                dummy_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(DATA_DIR, class_name, f'dummy_{i}.png'), dummy_image)

    train_and_evaluate()
