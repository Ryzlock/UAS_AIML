import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Konfigurasi
IMG_SIZE = (112, 112)
BATCH_SIZE = 32
EPOCHS = 128
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "data/cnn_face_model.h5"
LABEL_MAP_PATH = "data/label_map.json"

# Fungsi untuk melatih model CNN
def train_model(progress_bar, root):
    # Preprocessing data dengan ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Membuat model CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # update progress setiap epoch
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            percent = int((epoch + 1) / EPOCHS * 100)
            progress_bar['value'] = percent
            progress_bar.update()

    # train model
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[EarlyStopping(patience=5), ProgressCallback()]
    )

    # Simpan model dan label mapping
    model.save(MODEL_SAVE_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(train_gen.class_indices, f)

    # Tampilkan pesan selesai dan keluar setelah 2 detik
    messagebox.showinfo("Selesai", "Model berhasil dilatih dan disimpan.")
    root.after(2000, root.destroy)

# --- Setup GUI ---
root = tk.Tk()
root.title("Training Model")
root.geometry("500x200")

ttk.Label(root, text="Training model CNN").pack(pady=20)
progress = ttk.Progressbar(root, length=400, mode='determinate')
progress.pack(pady=10)

# Supaya GUI tidak freeze
thread = threading.Thread(target=train_model, args=(progress, root))
thread.start()

root.mainloop()
