# Good till now

import tkinter as tk
from tkinter import ttk
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
from tkinter import simpledialog, messagebox
import shutil
import numpy as np
import time
import dlib
from scipy.spatial import distance


# Ensure required directories exist
os.makedirs("known_faces", exist_ok=True)
os.makedirs("captured_faces", exist_ok=True)
os.makedirs("attendance_records", exist_ok=True)  # Directory for attendance files

# Global variables
# Global dictionary to store Roll Number to Name mapping
roll_number_name_mapping = {}

log_widget = None
subjects = ["Math", "Science", "History"]  # Subjects for attendance tracking

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize subject-wise Excel files
def initialize_excel():
    for subject in subjects:
        file_name = os.path.join("attendance_records", f"{subject}.xlsx")
        if not os.path.exists(file_name):
            df = pd.DataFrame(columns=["Roll Number"])
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

# Load known faces
def load_known_faces():
    global roll_number_name_mapping
    known_faces = []
    known_roll_numbers = {}

    # Ensure the dictionary is initialized before clearing
    if 'roll_number_name_mapping' not in globals():
        roll_number_name_mapping = {}

    roll_number_name_mapping.clear()

    for filename in os.listdir("known_faces"):
        image_path = os.path.join("known_faces", filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                known_faces.append(encoding)
                name, roll_number = os.path.splitext(filename)[0].split("_")
                known_roll_numbers[roll_number] = encoding
                roll_number_name_mapping[roll_number] = name
            else:
                log_message(f"Warning: No face detected in {filename}. Skipping.")
        except Exception as e:
            log_message(f"Error loading {filename}: {e}")

    return known_faces, known_roll_numbers


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


# Detect head movement
def detect_head_movement(landmarks, initial_nose):
    current_nose = (landmarks.part(30).x, landmarks.part(30).y)
    movement = abs(current_nose[0] - initial_nose[0]) + abs(current_nose[1] - initial_nose[1])
    return movement > 3  # Adjust threshold as needed


def is_blinking(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear < 0.2  # Lowered threshold for better accuracy

def capture_image():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        log_message("Error: Webcam not accessible.")
        return None

    log_message("Detecting liveness...")
    blink_count = 0
    blink_detected = False
    last_blink_time = time.time()
    eye_open = True
    start_time = time.time()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            log_message("No face detected. Adjust lighting or position...")
            continue

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            if is_blinking(left_eye) or is_blinking(right_eye):
                if eye_open:  # Only count a new blink when eyes were previously open
                    blink_count += 1
                    last_blink_time = time.time()
                    eye_open = False  # Mark eyes as closed
            else:
                eye_open = True  # Reset when eyes open again

        if time.time() - last_blink_time > 5:
            log_message("No live face detected (No blink in time). Liveness check failed.")
            video_capture.release()
            return None

        if blink_count >= 2:
            blink_detected = True
            break

    video_capture.release()
    filename = os.path.join("captured_faces", f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(filename, frame)
    log_message(f"Image captured and saved as {filename}")
    return filename

# Process captured image and update Excel
def process_image():
    known_faces, known_roll_numbers = load_known_faces()
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

    excel_file = os.path.join("attendance_records", f"{subject}.xlsx")
    df = pd.read_excel(excel_file)
    date_today = datetime.now().strftime('%Y-%m-%d')

    # Ensure column exists for today
    if date_today not in df.columns:
        df[date_today] = "A"

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(known_roll_numbers.values()), face_encoding)
        if True in matches:
            best_match_index = matches.index(True)
            roll_number = list(known_roll_numbers.keys())[best_match_index]
            name = roll_number_name_mapping.get(roll_number, "Unknown")

            log_message(f"Recognized: {name} (Roll No: {roll_number})")

            # Prevent Duplicate Entry for Today
            if roll_number in df["Roll Number"].astype(str).values:
                existing_entry = df.loc[df["Roll Number"].astype(str) == roll_number, date_today].values[0]
                if existing_entry == "P":
                    log_message(f"Duplicate entry detected for {name} (Roll No: {roll_number}) on {date_today}.")
                    return
                else:
                    df.loc[df["Roll Number"].astype(str) == roll_number, date_today] = "P"
            else:
                new_row = {"Roll Number": roll_number, "Name": name, date_today: "P"}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            log_message(f"Attendance marked for {name} (Roll No: {roll_number}) in {subject}")
        else:
            log_message("Unknown face detected.")

    df.to_excel(excel_file, index=False)
    log_message(f"Excel file updated for {subject}.")

def add_face():
    filename = capture_image()
    if not filename:
        return

    roll_number = simpledialog.askstring("Input", "Enter Roll Number:")
    name = simpledialog.askstring("Input", "Enter Student Name:")
    if not name or not roll_number:
        log_message("Operation canceled: Missing Roll Number or Name.")
        return

    # Ensure uniqueness
    known_faces, known_roll_numbers = load_known_faces()
    if roll_number in known_roll_numbers:
        log_message("Roll number already exists. Cannot add duplicate entries.")
        return

    # Save the face image
    new_path = os.path.join("known_faces", f"{name}_{roll_number}.jpg")
    shutil.move(filename, new_path)
    log_message(f"New face added as {new_path}")

    # Add the roll number and name to all attendance records
    for subject in subjects:
        file_name = os.path.join("attendance_records", f"{subject}.xlsx")
        df = pd.read_excel(file_name)

        if str(roll_number) not in df["Roll Number"].astype(str).values:
            df = pd.concat([df, pd.DataFrame([{"Roll Number": roll_number, "Name": name}])], ignore_index=True)
            df.to_excel(file_name, index=False)
            log_message(f"Added {name} (Roll No: {roll_number}) to {subject} attendance file.")

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
