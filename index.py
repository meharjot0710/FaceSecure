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

# Global dictionary to store Roll Number to Name mapping
roll_number_name_mapping = {}

# Global variables
log_widget = None
subjects = ["Math", "OS", "DSA"]  # Subjects for attendance tracking

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

    cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Camera", 400, 300)
    
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
                if eye_open:
                    blink_count += 1
                    last_blink_time = time.time()
                    eye_open = False
            else:
                eye_open = True

        if time.time() - last_blink_time > 3:
            log_message("No live face detected (No blink in time). Liveness check failed.")
            video_capture.release()
            cv2.destroyWindow("Live Camera")
            return None

        if blink_count >= 2:
            blink_detected = True
            break

        cv2.imshow("Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyWindow("Live Camera")
    filename = os.path.join("captured_faces", f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(filename, frame)
    log_message(f"Image captured and saved as {filename}")
    return filename

# Process captured image and update Excel
import os
import pandas as pd
import face_recognition
from datetime import datetime
from tkinter import simpledialog

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

    recognized_students = {}

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(known_roll_numbers.values()), face_encoding)
        if True in matches:
            best_match_index = matches.index(True)
            roll_number = list(known_roll_numbers.keys())[best_match_index]
            name = roll_number_name_mapping.get(roll_number, "Unknown")

            log_message(f"Recognized: {name} (Roll No: {roll_number})")
            recognized_students[roll_number] = name
        else:
            log_message("Unknown face detected.")

    if not recognized_students:
        log_message("No known faces recognized. Attendance not recorded.")
        return

    subject = simpledialog.askstring("Input", f"Select subject: {', '.join(subjects)}")
    if subject not in subjects:
        log_message("Invalid subject selection.")
        return

    excel_file = os.path.join("attendance_records", f"{subject}.xlsx")
    df = pd.read_excel(excel_file)
    date_today = datetime.now().strftime('%Y-%m-%d')

    if date_today not in df.columns:
        df[date_today] = "A"

    for roll_number, name in recognized_students.items():
        if roll_number in df["Roll Number"].astype(str).values:
            existing_entry = df.loc[df["Roll Number"].astype(str) == roll_number, date_today].values[0]
            if existing_entry == "P":
                log_message(f"Duplicate entry detected for {name} (Roll No: {roll_number}) on {date_today}.")
                continue
            else:
                df.loc[df["Roll Number"].astype(str) == roll_number, date_today] = "P"
        else:
            new_row = {"Roll Number": roll_number, "Name": name, date_today: "P"}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        log_message(f"Attendance marked for {name} (Roll No: {roll_number}) in {subject}")

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


def pp(return_recognized=False):
    known_faces, known_roll_numbers = load_known_faces()
    if not known_faces:
        log_message("No known faces found. Please add faces to the 'known_faces' folder.")
        return (None, None) if return_recognized else None

    captured_image = capture_image()
    if not captured_image:
        return (None, None) if return_recognized else None

    image = face_recognition.load_image_file(captured_image)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        log_message("No faces detected in the captured image.")
        return (None, None) if return_recognized else None

    recognized_students = {}

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(known_roll_numbers.values()), face_encoding)
        if True in matches:
            best_match_index = matches.index(True)
            roll_number = list(known_roll_numbers.keys())[best_match_index]
            name = roll_number_name_mapping.get(roll_number, "Unknown")

            log_message(f"Recognized: {name} (Roll No: {roll_number})")
            recognized_students[roll_number] = name
        else:
            log_message("Unknown face detected.")

    if return_recognized:
        if recognized_students:
            first_student = list(recognized_students.items())[0]  # Return first recognized student
            return first_student[1], first_student[0]  # Return (name, roll number)
        return (None, None)

    if not recognized_students:
        log_message("No known faces recognized. Attendance not recorded.")
        return None

def predict_attendance():
    recognized_name, recognized_roll = pp(return_recognized=True)
    
    if recognized_name is None or recognized_roll is None:
        messagebox.showerror("Error", "Face not recognized. Please try again.")
        return
    
    messagebox.showinfo("Face Recognized", f"Welcome, {recognized_name} (Roll No: {recognized_roll})")

    # Step 2: Ask for subject selection
    subject = simpledialog.askstring("Input", f"Select subject: {', '.join(subjects)}")
    if subject not in subjects:
        messagebox.showerror("Error", "Invalid subject selection.")
        return

    excel_file = os.path.join("attendance_records", f"{subject}.xlsx")
    
    if not os.path.exists(excel_file):
        messagebox.showerror("Error", f"No attendance record found for {subject}.")
        return
    
    # Load the Excel file
    df = pd.read_excel(excel_file)

    print("Loaded DataFrame:\n", df.head())  # Print first few rows
    print("Columns in Excel File:", df.columns)  # Print all column names

    if "Roll Number" not in df.columns:
        messagebox.showerror("Error", "Invalid Excel format. Missing 'Roll Number' column.")
        return

    # Strip spaces from column names (in case of formatting issues)
    df.columns = df.columns.str.strip()

    # Extract attendance columns
    date_columns = [col for col in df.columns if col not in ["Roll Number", "Name"]]
    print("Extracted Date Columns:", date_columns)

    total_classes = len(date_columns)

    if total_classes == 0:
        messagebox.showinfo("Info", "No classes conducted yet for prediction.")
        return

    # Step 3: Find the student's attendance record
    
    student_row = df[df["Roll Number"] == int(recognized_roll)]

    if student_row.empty:
        messagebox.showerror("Error", "No attendance record found for this student.")
        return

    student_row = student_row.iloc[0]

    # Debugging: Print student's attendance data
    print("Student Attendance Data:", student_row[date_columns].to_dict())

    # Debugging: Check unique attendance values
    print("Unique Attendance Values in Excel:", df[date_columns].stack().unique())

    # Fix attendance counting issue
    total_present = sum(str(student_row[date]).strip().upper() in ["P", "1"] for date in date_columns)
    print("Total Present Count:", total_present)

    # Projected future attendance
    remaining_classes = 50  # Assume 50 more classes
    projected_present = total_present + remaining_classes
    projected_percentage = (projected_present / (total_classes + remaining_classes)) * 100

    # Fix max missed classes calculation
    min_attendance_required = 75  # Minimum percentage required
    remaining_needed = (min_attendance_required / 100) * (total_classes + remaining_classes)
    max_missed = 50 - max(0, int(remaining_needed - total_present))

    prediction_result = (
        f"{recognized_name} (Roll No: {recognized_roll}):\n"
        f"Current Attendance: {total_present}/{total_classes} ({(total_present / total_classes) * 100:.2f}%)\n"
        f"Projected Attendance: {projected_present}/{total_classes + remaining_classes} ({projected_percentage:.2f}%)\n"
        f"You can miss {max_missed} more classes."
    )

    messagebox.showinfo("Attendance Prediction", prediction_result)


def setup_gui():
    global log_widget

    root = tk.Tk()
    root.title("FaceSecure")
    root.geometry("700x500")
    root.configure(bg="#2C3E50")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=10)
    style.configure("T`Label", font=("Arial", 14), background="#2C3E50", foreground="white")

    header_frame = tk.Frame(root, bg="#1A252F", pady=10)
    header_frame.pack(fill=tk.X)
    title_label = tk.Label(header_frame, text="FaceSecure", font=("Arial", 18, "bold"), bg="#1A252F", fg="white")
    title_label.pack()

    main_frame = tk.Frame(root, bg="#2C3E50", padx=20, pady=20)
    main_frame.pack(expand=True)

    button_frame = tk.Frame(main_frame, bg="#2C3E50")
    button_frame.pack(pady=10)
    
    ttk.Button(button_frame, text="Mark Attendance", command=process_image).grid(row=0, column=0, padx=15, pady=10)
    ttk.Button(button_frame, text="Add Face", command=add_face).grid(row=0, column=1, padx=15, pady=10)
    ttk.Button(button_frame, text="Attendance Prediction", command=predict_attendance).grid(row=1, column=0, columnspan=2, padx=15, pady=10)
    
    log_label = ttk.Label(main_frame, text="Logs:")
    log_label.pack(anchor="w", pady=(20, 5))

    log_widget = tk.Text(main_frame, width=80, height=15, wrap="word", bg="#ECF0F1", fg="#2C3E50", font=("Arial", 10))
    log_widget.pack()

    root.mainloop()

if __name__ == "__main__":
    initialize_excel()
    setup_gui()
