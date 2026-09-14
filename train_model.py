"""
train_model.py
Trains a CNN on Quick, Draw! bitmap data (28×28 grayscale).
Saves the trained model and category list for inference.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Configuration ───────────────────────────────────────────────────
SAMPLES_PER_CATEGORY = 5000      # Heavy model Training
IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 20
VALIDATION_SPLIT = 0.2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.json")


def load_data():
    """Load consolidated .npy files and prepare training data."""
    if not os.path.exists(CATEGORIES_PATH):
        raise FileNotFoundError(f"Missing {CATEGORIES_PATH}. Run update_categories_json.py!")

    with open(CATEGORIES_PATH, "r") as f:
        categories_dict = json.load(f)

    print("\n[*] Loading consolidated data ...")
    X_all = []
    y_all = []
    
    current_idx = 0
    for pack in ["1", "2", "3"]:
        filepath = os.path.join(DATA_DIR, f"set{pack}.npy")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing {filepath}. Run data_prepper.py first!")

        # Load the set (contains many categories stacked)
        set_data = np.load(filepath)
        
        # We need to create labels for each sample in the set
        # data_prepper.py stacks them in order of categories in categories_dict[pack]
        pack_categories = categories_dict[pack]
        num_categories = len(pack_categories)
        samples_in_set = set_data.shape[0]
        
        # Each category has SAMPLES_PER_CATEGORY (though some might have failed, we assume the prepper worked)
        # However, it's safer to calculate based on the actual shape
        samples_per_cat_actual = samples_in_set // num_categories
        
        X_all.append(set_data)
        for i in range(num_categories):
            y_all.append(np.full(samples_per_cat_actual, current_idx))
            current_idx += 1
            
        print(f"  [OK] Set {pack}: {samples_in_set} samples ({num_categories} categories)")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # Reshape from flat (784,) → (28, 28, 1) and normalise to [0, 1]
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

    print(f"\n  Total samples : {X.shape[0]}")
    print(f"  Image shape   : {X.shape[1:]}")
    print(f"  Num categories: {current_idx}")

    return X, y, current_idx


def build_model(num_classes: int) -> keras.Model:
    """Build a more powerful CNN for 50-category sketch recognition."""
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        data_augmentation,
        # ── Block 1: 64 filters ──
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # ── Block 2: 128 filters ──
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # ── Block 3: 256 filters ──
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # ── Classifier ──
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    """Main training pipeline."""
    X, y, num_classes = load_data()

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    print(f"\n  Train set: {X_train.shape[0]} | Val set: {X_val.shape[0]}")

    # Build model
    model = build_model(num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    # Train
    print("\n[*] Training ...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'=' * 50}")
    print(f"[OK] Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"     Validation Loss     : {val_loss:.4f}")
    print(f"{'=' * 50}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "sketch_model.keras")
    model.save(model_path)
    print(f"\n[SAVED] Model saved to: {model_path}")

    # categories.json is already managed by update_categories_json.py
    # but we print its location to be consistent.
    print(f"[INFO] Categories file: {CATEGORIES_PATH}")

    return history


if __name__ == "__main__":
    print("=" * 50)
    print("  Quick, Draw! CNN Trainer")
    print("=" * 50)
    train()
