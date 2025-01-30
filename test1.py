import tkinter as tk
from tkinter import ttk
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
from tkinter import simpledialog, messagebox
import shutil

# Ensure required directories exist
os.makedirs("known_faces", exist_ok=True)
os.makedirs("captured_faces", exist_ok=True)

# Global variables
log_widget = None
subjects = ["Math", "Science", "History"]  # Subjects for attendance tracking

# Initialize subject-wise Excel files
def initialize_excel():
    for subject in subjects:
        file_name = f"{subject}.xlsx"
        if not os.path.exists(file_name):
            df = pd.DataFrame(columns=["Roll Number", "Name"])
            df.to_excel(file_name, index=False)
            log_message(f"Initialized new Excel file: {file_name}")

# Log message function
def log_message(message):
    global log_widget
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = f"[{timestamp}] {message}"
    log_widget.insert(tk.END, log + '\n')
    log_widget.yview(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(log + "\n")

# Ensure captured_faces folder has at most 2 images
def manage_captured_faces():
    files = sorted(
        [os.path.join("captured_faces", f) for f in os.listdir("captured_faces") if f.endswith(".jpg")],
        key=os.path.getctime
    )
    while len(files) > 2:
        os.remove(files[0])
        log_message(f"Deleted old image: {files[0]}")
        files.pop(0)

# Load known faces
def load_known_faces():
    known_faces = []
    known_names = []
    known_roll_numbers = {}

    for filename in os.listdir("known_faces"):
        image_path = os.path.join("known_faces", filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            name, roll_number = os.path.splitext(filename)[0].split("_")
            known_names.append(name)
            known_roll_numbers[roll_number] = name
        except Exception as e:
            log_message(f"Error loading {filename}: {e}")

    return known_faces, known_names, known_roll_numbers

# Capture image from webcam
def capture_image():
    manage_captured_faces()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        log_message("Error: Webcam not accessible.")
        return None

    log_message("Capturing image...")
    ret, frame = video_capture.read()
    video_capture.release()

    if ret:
        filename = os.path.join("captured_faces", f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        log_message(f"Image captured and saved as {filename}")
        return filename
    else:
        log_message("Error: Failed to capture image.")
        return None

# Process captured image and update Excel
def process_image():
    known_faces, known_names, known_roll_numbers = load_known_faces()
    if not known_faces:
        log_message("No known faces found. Please add faces to the 'known_faces' folder.")
        return

    captured_image = capture_image()
    if not captured_image:
        return

    image = face_recognition.load_image_file(captured_image)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        log_message("No faces detected in the captured image.")
        return

    subject = simpledialog.askstring("Input", f"Select subject: {', '.join(subjects)}")
    if subject not in subjects:
        log_message("Invalid subject selection.")
        return

    excel_file = f"{subject}.xlsx"
    df = pd.read_excel(excel_file)

    date_today = datetime.now().strftime('%Y-%m-%d')

    # Ensure column exists for today
    if date_today not in df.columns:
        df[date_today] = "A"

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        if True in matches:
            best_match_index = matches.index(True)
            name = known_names[best_match_index]
            roll_number = [key for key, value in known_roll_numbers.items() if value == name][0]

            log_message(f"Recognized: {name} (Roll No: {roll_number})")

            # **Fix 1: Ensure roll number is treated as string for consistency**
            roll_number = str(roll_number)

            # **Fix 2: Prevent Duplicate Entry for Today**
            if roll_number in df["Roll Number"].astype(str).values:
                existing_entry = df.loc[df["Roll Number"].astype(str) == roll_number, date_today].values[0]
                if existing_entry == "P":
                    log_message(f"Duplicate entry detected for {name} (Roll No: {roll_number}) on {date_today}.")
                    return
                else:
                    df.loc[df["Roll Number"].astype(str) == roll_number, date_today] = "P"
            else:
                # **Fix 3: Ensure new entry is appended correctly**
                new_row = {"Roll Number": roll_number, "Name": name, date_today: "P"}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            log_message(f"Attendance marked for {name} (Roll No: {roll_number}) in {subject}")
        else:
            log_message("Unknown face detected.")

    # **Fix 4: Ensure Excel file updates correctly**
    df.to_excel(excel_file, index=False)
    log_message(f"Excel file updated for {subject}.")

# Add face to known faces
def add_face():
    filename = capture_image()
    if not filename:
        return

    roll_number = simpledialog.askstring("Input", "Enter Roll Number:")
    name = simpledialog.askstring("Input", "Enter Student Name:")
    if not name or not roll_number:
        log_message("Operation canceled: Missing Roll Number or Name.")
        return

    known_faces, known_names, known_roll_numbers = load_known_faces()
    if roll_number in known_roll_numbers:
        log_message("Roll number already exists. Cannot add duplicate entries.")
        return

    new_path = os.path.join("known_faces", f"{name}_{roll_number}.jpg")
    shutil.move(filename, new_path)
    log_message(f"New face added as {new_path}")

    for subject in subjects:
        excel_file = f"{subject}.xlsx"
        df = pd.read_excel(excel_file, index_col=0)
        if roll_number not in df.index:
            df.loc[int(roll_number)] = [name] + ["A"] * (len(df.columns) - 1)
            df.to_excel(excel_file)
            log_message(f"New student {name} (Roll No: {roll_number}) added to {subject} attendance sheet.")

# GUI Setup
def setup_gui():
    global log_widget

    root = tk.Tk()
    root.title("Face Recognition System")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Button(main_frame, text="Capture & Process", command=process_image).grid(row=0, column=0, padx=5, pady=5)
    ttk.Button(main_frame, text="Add Face", command=add_face).grid(row=0, column=1, padx=5, pady=5)

    log_widget = tk.Text(main_frame, width=80, height=20, wrap="word")
    log_widget.grid(row=1, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    initialize_excel()
    setup_gui()
