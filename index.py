import tkinter as tk
from tkinter import ttk
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
from tkinter import simpledialog
import shutil

# Ensure required directories exist
os.makedirs("known_faces", exist_ok=True)
os.makedirs("captured_faces", exist_ok=True)

# Global variables
log_widget = None
excel_file = "Data.xlsx"

# Initialize Excel file
def initialize_excel():
    if not os.path.exists(excel_file):
        df = pd.DataFrame(columns=["Date", "Name", "Entry Time", "Exit Time"])
        df.to_excel(excel_file, index=False)
        log_message("Initialized new Excel file: Data.xlsx")

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
    known_faces = []
    known_names = []

    for filename in os.listdir("known_faces"):
        image_path = os.path.join("known_faces", filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])
        except Exception as e:
            log_message(f"Error loading {filename}: {e}")

    return known_faces, known_names

# Capture image from webcam
def capture_image():
    # Directory to store captured images
    directory = "captured_faces"
    # Check if the directory exists and has more than 5 images
    files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg")],
        key=os.path.getctime
    )
    if len(files) > 4:
        # Delete oldest images to maintain only 4 images
        for file_to_delete in files[:-5]:
            os.remove(file_to_delete)
            log_message(f"Deleted old image: {file_to_delete}")

    # Capture a new image
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        log_message("Error: Webcam not accessible.")
        return None

    log_message("Capturing image...")
    ret, frame = video_capture.read()
    video_capture.release()

    if ret:
        filename = os.path.join(directory, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        log_message(f"Image captured and saved as {filename}")
        return filename
    else:
        log_message("Error: Failed to capture image.")
        return None

# Process captured image and update Excel
def process_image():
    known_faces, known_names = load_known_faces()
    if not known_faces:
        log_message("No known faces found. Please add faces to the 'known_faces' folder.")
        return

    captured_image = capture_image()
    if not captured_image:
        return

    image = face_recognition.load_image_file(captured_image)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        log_message("No faces detected in the captured image.")
        return

    df = pd.read_excel(excel_file)
    if "Date" not in df.columns:
        df["Date"] = None
    date_today = datetime.now().strftime('%Y-%m-%d')

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        if True in matches:
            best_match_index = matches.index(True)
            name = known_names[best_match_index]
            log_message(f"Recognized: {name}")

            # Check if entry already exists
            existing_entry = df[(df["Date"] == date_today) & (df["Name"] == name)]
            if existing_entry.empty:
                entry_time = datetime.now().strftime('%H:%M:%S')
                new_row = {"Date": date_today, "Name": name, "Entry Time": entry_time, "Exit Time": ""}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                log_message(f"Entry time recorded for {name}: {entry_time}")
            else:
                exit_time = datetime.now().strftime('%H:%M:%S')
                df.loc[existing_entry.index, "Exit Time"] = exit_time
                log_message(f"Exit time recorded for {name}: {exit_time}")
        else:
            log_message("Unknown face detected.")

    df.to_excel(excel_file, index=False)
    log_message("Excel file updated.")

# Add face to known faces
def add_face():
    filename = capture_image()
    if not filename:
        return

    name = simpledialog.askstring("Input", "Enter the name for the captured face:")
    if not name:
        log_message("Operation canceled: No name provided.")
        return

    new_path = os.path.join("known_faces", f"{name}.jpg")
    shutil.move(filename, new_path)
    log_message(f"New face added as {new_path}")

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
