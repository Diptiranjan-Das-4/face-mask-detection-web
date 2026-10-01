import cv2
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with_mask_path = os.path.join(
    BASE_DIR,
    "maskdata",
    "maskdata",
    "train",
    "with_mask"
)

without_mask_path = os.path.join(
    BASE_DIR,
    "maskdata",
    "maskdata",
    "train",
    "without_mask"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 250

data = []
labels = []


# ============================================================
# LOAD WITH-MASK IMAGES
# ============================================================

print("Loading with_mask images...")

for file in os.listdir(with_mask_path):

    img_path = os.path.join(with_mask_path, file)

    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)


# ============================================================
# LOAD WITHOUT-MASK IMAGES
# ============================================================

print("Loading without_mask images...")

for file in os.listdir(without_mask_path):

    img_path = os.path.join(without_mask_path, file)

    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)


# ============================================================
# CHECK DATA
# ============================================================

print()
print("With mask images:", labels.count(1))
print("Without mask images:", labels.count(0))
print("Total images:", len(labels))

if len(data) == 0:
    raise RuntimeError("No images were found in the dataset.")


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(data) / 255.0
y = np.array(labels)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print()
print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ============================================================
# CREATE CNN
# SAME ARCHITECTURE AS YOUR ORIGINAL NOTEBOOK
# ============================================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Starting training...")
print()

model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)


# ============================================================
# TEST ACCURACY
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print()


# ============================================================
# SAVE NEW MODEL
# ============================================================

new_model_path = os.path.join(
    BASE_DIR,
    "mask_detector_model_new.h5"
)

model.save(new_model_path)

print("New model saved:")
print(new_model_path)
print()