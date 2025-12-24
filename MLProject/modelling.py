import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mlflow
import mlflow.tensorflow

DATA_PATH = "dataset_bunga_clean"

# Konfigurasi MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Eksperimen_Bunga")

def load_data():
    X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'), allow_pickle=True)
    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    mlflow.tensorflow.autolog()
    
    print("Mulai Training dengan Autolog...")

    model = build_model(input_shape=(128, 128, 3), num_classes=4)
    
    model.fit(X_train, y_train, 
              epochs=5, 
              validation_data=(X_test, y_test),
              batch_size=32)
    
    print("Training selesai. Cek artifact di MLflow.")
