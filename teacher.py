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
import ttkbootstrap as ttkb
from PIL import Image, ImageTk

ATTENDANCE_FOLDER = "attendance_records"
WARNING_FILE = "warnings.xlsx"

def get_subjects():
    files = [f.replace(".xlsx", "") for f in os.listdir(ATTENDANCE_FOLDER) if f.endswith(".xlsx")]
    return files

def load_attendance():
    selected_subject = subject_var.get()
    file_path = os.path.join(ATTENDANCE_FOLDER, f"{selected_subject}.xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        total_classes = len(df.columns) - 2
        df['Attendance %'] = (df.iloc[:, 2:].apply(lambda row: (row == 'P').sum(), axis=1) / total_classes) * 100
        for i in tree.get_children():
            tree.delete(i)
        for _, row in df.iterrows():
            tree.insert("", "end", values=(row['Roll Number'], row['Name'], f"{row['Attendance %']:.2f}%"))
    else:
        messagebox.showerror("Error", "File not found!")

face_detect_faces = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_detect_faces.setInput(blob)
    detections = face_detect_faces.forward()
    faces = []
    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def log_message(message):
    global log_widget
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = f"[{timestamp}] {message}"
    log_widget.insert(tk.END, log + '\n')
    log_widget.yview(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(log + "\n")

def manage_captured_faces():
    files = sorted(
        [os.path.join("captured_faces", f) for f in os.listdir("captured_faces") if f.endswith(".jpg")],
        key=os.path.getctime
    )
    while len(files) > 2:
        os.remove(files[0])
        log_message(f"Deleted old image: {files[0]}")
        files.pop(0)

def detect_head_movement(landmarks, initial_nose):
    current_nose = (landmarks.part(30).x, landmarks.part(30).y)
    movement = abs(current_nose[0] - initial_nose[0]) + abs(current_nose[1] - initial_nose[1])
    return movement > 3

def is_blinking(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear < 0.2

def load_known_faces():
    global roll_number_name_mapping
    known_faces = []
    known_roll_numbers = {}
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

def get_face_encodings(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_img, model="cnn")
    encodings = face_recognition.face_encodings(rgb_img, faces)
    return encodings, faces

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
        faces = detect_faces(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(faces) == 0:
            log_message("No face detected. Adjust lighting or position...")
            continue
        for face in faces:
            dlib_rect = dlib.rectangle(face[0], face[1], face[2], face[3])
            landmarks = predictor(gray, dlib_rect)
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

def match_face(known_encodings, face_encoding, known_roll_numbers):
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < 0.5:
        roll_number = list(known_roll_numbers.keys())[best_match_index]
        return roll_number
    return None

def process_image():
    known_faces, known_roll_numbers = load_known_faces()
    if not known_faces:
        log_message("No known faces found. Please add faces to the 'known_faces' folder.")
        return
    captured_image = capture_image()
    if not captured_image:
        return
    image = face_recognition.load_image_file(captured_image)
    get_face_encodings = face_recognition.face_encodings(image)
    if not get_face_encodings:
        log_message("No faces detected in the captured image.")
        return
    recognized_students = {}
    for face_encoding in get_face_encodings:
        roll_number = match_face(list(known_roll_numbers.values()), face_encoding, known_roll_numbers)
        if roll_number is not None:
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

def get_student_info():
    dialog = tk.Toplevel()
    dialog.title("Student Info")
    dialog.geometry("300x200")
    tk.Label(dialog, text="Enter Roll Number:").pack(pady=5)
    roll_number_entry = tk.Entry(dialog)
    roll_number_entry.pack(pady=5)
    tk.Label(dialog, text="Enter Student Name:").pack(pady=5)
    name_entry = tk.Entry(dialog)
    name_entry.pack(pady=5)
    def submit():
        roll_number = roll_number_entry.get()
        name = name_entry.get()
        if roll_number and name:
            dialog.result = (roll_number, name)
            dialog.destroy()
        else:
            tk.Label(dialog, text="Both fields are required!", fg="red").pack()
    submit_btn = tk.Button(dialog, text="Submit", command=submit)
    submit_btn.pack(pady=10)
    dialog.result = None
    dialog.grab_set()
    dialog.wait_window()
    return dialog.result

def add_face():
    filename = capture_image()
    if not filename:
        return
    known_faces, known_roll_numbers = load_known_faces()
    image = face_recognition.load_image_file(filename)
    get_face_encodings = face_recognition.face_encodings(image)
    recognized_students = {}
    for face_encoding in get_face_encodings:
        roll_number = match_face(list(known_roll_numbers.values()), face_encoding, known_roll_numbers)
        if roll_number is not None:
            name = roll_number_name_mapping.get(roll_number, "Unknown")
            log_message(f"Recognized: {name} (Roll No: {roll_number})")
            log_message(f"Your face is already registered")
            recognized_students[roll_number] = name
    if bool(recognized_students):
        return
    if not get_face_encodings:
        log_message("No faces detected in the captured image.")
        return
    student_info = get_student_info()
    if not student_info:
        log_message("Operation canceled: Missing Roll Number or Name.")
        return
    roll_number, name = student_info
    print(roll_number)
    print(name)
    if roll_number in known_roll_numbers:
        print("hhh")
        log_message("Roll number already exists. Cannot add duplicate entries.")
        return
    new_path = os.path.join("known_faces", f"{name}_{roll_number}.jpg")
    shutil.move(filename, new_path)
    log_message(f"New face added as {new_path}")
    for subject in subjects:
        file_name = os.path.join("attendance_records", f"{subject}.xlsx")
        df = pd.read_excel(file_name)
        if str(roll_number) not in df["Roll Number"].astype(str).values:
            df = pd.concat([df, pd.DataFrame([{"Roll Number": roll_number, "Name": name}])], ignore_index=True)
            df.to_excel(file_name, index=False)
            log_message(f"Added {name} (Roll No: {roll_number}) to {subject} attendance file.")

def send_warnings():
    selected_subject = subject_var.get()
    file_path = os.path.join(ATTENDANCE_FOLDER, f"{selected_subject}.xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        total_classes = len(df.columns) - 2
        df['Attendance %'] = (df.iloc[:, 2:].apply(lambda row: (row == 'P').sum(), axis=1) / total_classes) * 100
        warnings = df[df['Attendance %'] < 75][['Roll Number', 'Name']].copy()
        warnings["Subject"] = selected_subject
        warnings["Message"] = "Your attendance is below 75%. Please improve it."
        if os.path.exists(WARNING_FILE):
            existing_warnings = pd.read_excel(WARNING_FILE)
            existing_warnings = existing_warnings[['Roll Number', 'Name', 'Subject', 'Message']]
            new_warnings = warnings.merge(existing_warnings, on=['Roll Number', 'Subject'], how='left', indicator=True)
            new_warnings = new_warnings[new_warnings['_merge'] == 'left_only'].drop(columns=['_merge'])
            if not new_warnings.empty:
                updated_warnings = pd.concat([existing_warnings, new_warnings], ignore_index=True)
                updated_warnings.to_excel(WARNING_FILE, index=False)
                messagebox.showinfo("Warnings Sent", "Warnings have been sent to students with attendance below 75%.")
            else:
                messagebox.showinfo("No New Warnings", "No new students need warnings for this subject.")
        else:
            warnings.to_excel(WARNING_FILE, index=False)
            messagebox.showinfo("Warnings Sent", "Warnings have been sent to students with attendance below 75%.")
    else:
        messagebox.showerror("Error", "File not found!")

def update_attendance_criteria(roll_number):
    for subject in subjects:
        file_name = os.path.join("attendance_records", f"{subject}.xlsx")
        df = pd.read_excel(file_name)
        student_row = df[df["Roll Number"] == str(roll_number)]
        if not student_row.empty:
            attendance_percentage = calculate_attendance_percentage(student_row)
            if attendance_percentage < 75:
                new_criteria = 65
                messagebox.showinfo("Attendance Criteria Updated", f"Attendance criteria for Roll No: {roll_number} updated to 65%.")
            else:
                new_criteria = 75
            df.loc[df["Roll Number"] == str(roll_number), "Attendance Criteria"] = new_criteria
            df.to_excel(file_name, index=False)

def calculate_attendance_percentage(student_row):
    total_classes = student_row.shape[1] - 2
    total_present = sum(student_row.iloc[0][2:] == "P")
    return (total_present / total_classes) * 100

def review_leave_requests():
    leave_file = "leave_requests.xlsx"
    if not os.path.exists(leave_file):
        messagebox.showinfo("No Leave Requests", "No leave requests available.")
        return
    df = pd.read_excel(leave_file)
    pending_requests = df[df["Status"] == "Pending"]
    if pending_requests.empty:
        messagebox.showinfo("No Pending Requests", "No pending leave requests.")
        return
    review_window = tk.Toplevel()
    review_window.title("Review Leave Requests")
    review_window.geometry("800x400")
    columns = ("Roll Number", "Name", "Start Date", "End Date", "Reason", "Status")
    tree = ttk.Treeview(review_window, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=120)
    for index, row in pending_requests.iterrows():
        tree.insert("", "end", values=(row["Roll Number"], row["Name"], row["Start Date"], row["End Date"], row["Reason"], row["Status"]))
    tree.pack(pady=10, padx=10, fill="both", expand=True)
    def update_leave_status(status):
        selected_item = tree.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a leave request.")
            return
        item = tree.item(selected_item)
        roll_number = item["values"][0]
        df.loc[df["Roll Number"] == roll_number, "Status"] = status
        df.to_excel(leave_file, index=False)
        if status == "Accepted":
            update_attendance_criteria(roll_number)
        tree.delete(selected_item)
        messagebox.showinfo("Success", f"Leave for Roll No {roll_number} has been {status.lower()}.")
    btn_frame = tk.Frame(review_window)
    btn_frame.pack(pady=10)
    approve_btn = tk.Button(btn_frame, text="Approve", bg="green", fg="white", command=lambda: update_leave_status("Accepted"))
    approve_btn.pack(side="left", padx=10)
    reject_btn = tk.Button(btn_frame, text="Reject", bg="red", fg="white", command=lambda: update_leave_status("Rejected"))
    reject_btn.pack(side="right", padx=10)
    review_window.mainloop()

root = ttkb.Window(title="FaceSecure", themename="flatly")
root.geometry("1100x700")

header_frame = ttk.Frame(root, padding=20)
header_frame.pack(fill="x")

logo_path = "img3.jpg"
logo_img = Image.open(logo_path)
logo_img = logo_img.resize((60, 60), Image.Resampling.LANCZOS)
logo_tk = ImageTk.PhotoImage(logo_img)
logo_label = ttk.Label(header_frame, image=logo_tk)
logo_label.pack(side="left", padx=(0, 10))

title_label = ttk.Label(header_frame, text="FaceSecure", font=("Arial", 20, "bold"))
title_label.pack(side="left")

button_frame = ttk.Frame(root, padding=20)
button_frame.pack(fill="x")

btn_style = {"bootstyle": "primary", "width": 15, "padding": 10}

add_face_btn = ttk.Button(button_frame, text="Add Face", command=add_face, **btn_style)
add_face_btn.grid(row=0, column=0, padx=10)

review_leave_btn = ttk.Button(button_frame, text="Review Leave", command=review_leave_requests, **btn_style)
review_leave_btn.grid(row=0, column=1, padx=10)

subject_var = tk.StringVar()
subjects = get_subjects()

subject_dropdown = ttk.Combobox(button_frame, textvariable=subject_var, values=subjects, state='readonly', width=15)
subject_dropdown.grid(row=0, column=2, padx=10)
subject_dropdown.current(0)

show_attendance_btn = ttk.Button(button_frame, text="Show Attendance", command=load_attendance, **btn_style)
show_attendance_btn.grid(row=0, column=3, padx=10)

send_warnings_btn = ttk.Button(button_frame, text="Send Warnings", command=send_warnings, **btn_style)
send_warnings_btn.grid(row=0, column=4, padx=10)

attendance_card = ttk.Frame(root, padding=20)
attendance_card.pack(fill="both", expand=True)

attendance_label = ttk.Label(attendance_card, text="Attendance Records", font=("Arial", 16, "bold"))
attendance_label.pack(anchor="w")

table_frame = ttk.Frame(attendance_card)
table_frame.pack(fill="both", expand=True, pady=10)

columns = ("Roll Number", "Name", "Attendance %")
tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
for col in columns:
    tree.heading(col, text=col, anchor="center")
    tree.column(col, width=200, anchor="center")

tree.pack(fill="both", expand=True)

log_card = ttk.Frame(root, padding=20)
log_card.pack(fill="both", expand=True)

log_label = ttk.Label(log_card, text="Activity Log", font=("Arial", 16, "bold"))
log_label.pack(anchor="w")

log_widget = tk.Text(log_card, width=80, height=10, wrap="word", font=("Arial", 10))
log_widget.pack(fill="both", expand=True, pady=10)

root.mainloop()