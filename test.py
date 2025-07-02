import os
import cv2
import json
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import tkinter as tk
from tkinter import messagebox, Label, Entry, Button
from tkcalendar import DateEntry
from datetime import datetime
import time

# GUI untuk input kelas dan tanggal
def get_attendance_info():
    def submit():
        nonlocal class_name, class_date
        class_name = entry_class.get().strip().replace(" ", "_")
        class_date = calendar.get_date().strftime("%Y-%m-%d")
        if not class_name:
            tk.messagebox.showerror("Error", "Nama kelas harus diisi.")
            return
        root.destroy()

    class_name = ""
    class_date = ""

    root = tk.Tk()
    root.title("Input Info Absensi")

    # Ukuran window dan center
    window_width = 500
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    frame = tk.Frame(root)
    frame.pack(expand=True)

    tk.Label(frame, text="Nama Kelas:", font=("Arial", 12)).pack(pady=(20, 5))
    entry_class = tk.Entry(frame, font=("Arial", 12), width=30)
    entry_class.pack()

    tk.Label(frame, text="Tanggal:", font=("Arial", 12)).pack(pady=(20, 5))
    calendar = DateEntry(frame, width=27, font=("Arial", 12), background='darkblue',
                         foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
    calendar.pack()

    tk.Button(frame, text="Mulai Absensi", font=("Arial", 12), width=20, command=submit).pack(pady=30)

    root.mainloop()
    return class_name, class_date

# Konfigurasi
MODEL_PATH = "data/cnn_face_model.h5"
LABEL_MAP_PATH = "data/label_map.json"
USER_INFO_PATH = "data/user_info.csv"
IMG_SIZE = (112, 112)
THRESHOLD = 0.9
tf.get_logger().setLevel('ERROR')

# Load model dan label
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Load info user dari CSV
user_info = {}
with open(USER_INFO_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_info[row["NRP"]] = {"nama": row["Nama"], "kelamin": row["Kelamin"]}

# Setup file absensi
class_name, class_date = get_attendance_info()
attendance_file = f"Attendance/attendance_{class_name}_{class_date}.csv"
os.makedirs("Attendance", exist_ok=True)

if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["NRP", "Nama", "Kelamin", "Time"])

logged = set()
detected_times = {}

# Webcam dan face detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(f"Absensi dimulai untuk {class_name} pada tanggal {class_date} — Tekan 'q' untuk stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, IMG_SIZE)
        face_input = np.expand_dims(face_resized / 255.0, axis=0)

        preds = model.predict(face_input, verbose=0)[0]
        conf = float(np.max(preds))
        class_idx = int(np.argmax(preds))
        nrp = inv_label_map[class_idx]

        now = time.time()

        if conf >= THRESHOLD:
            name = user_info.get(nrp, {}).get("nama", "Tidak Dikenal")
            gender = user_info.get(nrp, {}).get("kelamin", "?")
            confidence_text = f"{int(conf * 100)}%"

            # Mulai hitung waktu deteksi
            if nrp not in detected_times:
                detected_times[nrp] = now

            time_diff = now - detected_times[nrp]

            if nrp not in logged and time_diff >= 5:
                with open(attendance_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([nrp, name, gender, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                logged.add(nrp)
                print(f"{nrp} - {name} Hadir")
                color = (0, 255, 0)
                status_text = ""
            elif nrp in logged:
                color = (0, 255, 255)
                status_text = " (Sudah Absen)"
            else:
                color = (255, 255, 0)
                status_text = f" (Tunggu {int(5 - time_diff)}s)"

            # Gambar kotak wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Nama di atas kotak
            cv2.putText(frame, f"{name}{status_text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # NRP dan Kelamin di bawah kotak
            label_bawah = f"{nrp} - {gender} | {confidence_text}"
            (text_w, text_h), _ = cv2.getTextSize(label_bawah, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y + h + 5), (x + text_w, y + h + 5 + text_h + 4), (0, 0, 0), -1)
            cv2.putText(frame, label_bawah, (x, y + h + text_h + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            label_text = f"Tidak Dikenal | {int(conf * 100)}%"
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
