import cv2
import os
import csv
import tkinter as tk
from tkinter import messagebox

# Konfigurasi
DATASET_DIR = "dataset"
USER_INFO_FILE = "data/user_info.csv"
IMG_SIZE = (112, 112)
NUM_IMAGES = 500

# Buat folder jika belum ada
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Fungsi utama capture wajah
def start_capture(nrp, name, gender):
    # Simpan info pengguna
    if not os.path.exists(USER_INFO_FILE):
        with open(USER_INFO_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["NRP", "Nama", "Kelamin"])
    with open(USER_INFO_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([nrp, name, gender])

    # Folder penyimpanan gambar wajah
    save_path = os.path.join(DATASET_DIR, nrp)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    messagebox.showinfo("Info", f"Proses capture untuk {name} dimulai.\nTekan 'q' untuk berhenti.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, IMG_SIZE)
            filename = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(filename, face_img)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) == ord('q') or count >= NUM_IMAGES:
            break

    cap.release()
    cv2.destroyAllWindows()

    messagebox.showinfo("Selesai", f"{count} gambar disimpan di: {save_path}")

# Fungsi saat tombol Submit ditekan
def submit():
    nrp = entry_nrp.get().strip()
    name = entry_name.get().strip()
    gender = entry_gender.get().strip().upper()

    if not nrp or not name or gender not in ("L", "P"):
        messagebox.showerror("Error", "Semua kolom wajib diisi dan kelamin hanya L/P.")
        return

    root.destroy()
    start_capture(nrp, name, gender)

# GUI Input Data
root = tk.Tk()
root.title("Registrasi Wajah Mahasiswa")

# Ukuran dan posisi tengah
window_width = 500
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Frame Tengah
frame = tk.Frame(root)
frame.pack(expand=True)

tk.Label(frame, text="NRP:", font=("Arial", 12)).pack(pady=5)
entry_nrp = tk.Entry(frame, font=("Arial", 12), width=30)
entry_nrp.pack()

tk.Label(frame, text="Nama:", font=("Arial", 12)).pack(pady=5)
entry_name = tk.Entry(frame, font=("Arial", 12), width=30)
entry_name.pack()

tk.Label(frame, text="Kelamin (L/P):", font=("Arial", 12)).pack(pady=5)
entry_gender = tk.Entry(frame, font=("Arial", 12), width=30)
entry_gender.pack()

tk.Button(frame, text="Mulai Capture", font=("Arial", 12), command=submit).pack(pady=20)

root.mainloop()
