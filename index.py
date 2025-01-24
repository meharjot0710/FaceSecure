import os
import face_recognition
import cv2
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from openpyxl import load_workbook
import tkinter as tk
from tkinter import messagebox, scrolledtext

log_widget = None

def log_message(message):
    global log_widget
    log_widget.insert(tk.END, message + '\n')
    log_widget.yview(tk.END) 

def get_excel_filename():
    return "Data.xlsx"

def manage_old_files():
    today = datetime.now().date()
    cutoff_date = today - timedelta(days=7)
    for filename in os.listdir('.'):
        if filename == "Data.xlsx":
            try:
                df = pd.read_excel(filename)
                dates = [col for col in df.columns if col.startswith('Entry Time_') or col.startswith('Exit Time_')]
                date_cols = [datetime.strptime(col.split('_')[2], '%Y-%m-%d').date() for col in dates if len(col.split('_')) > 2]
                if date_cols and any(date < cutoff_date for date in date_cols):
                    os.remove(filename)
                    log_message("Removed old data file due to exceeding 7-day limit.")
            except ValueError:
                continue

def on_entry_button_click():
    save_directory = "t"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    def calculate_brightness(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image)

    brightness_threshold = 100

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log_message("Error: Could not open webcam.")
        return

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            log_message("Error: Could not read frame.")
            break

        brightness = calculate_brightness(frame)
        face_locations = face_recognition.face_locations(frame)

        if face_locations and brightness > brightness_threshold:
            image_filename = os.path.join(save_directory, "captured_image.jpg")
            cv2.imwrite(image_filename, frame)
            log_message(f"Face detected. Image saved at {image_filename}")
            break

        if time.time() - start_time > 10:
            log_message("Exiting loop due to timeout.")
            break

    cap.release()

    folder_path = "t"
    known_faces_path = "known_faces"
    data = []

    known_faces = []
    known_names = []

    for filename in os.listdir(known_faces_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    today = datetime.now().date()
    excel_file = get_excel_filename()

    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        existing_names = existing_df["Identified Person"].tolist()
        if "Identified Person Number" in existing_df.columns:
            person_count = existing_df["Identified Person Number"].max() + 1 if not existing_df.empty else 1
        else:
            person_count = 1
    else:
        existing_df = pd.DataFrame()
        existing_names = []
        person_count = 1

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                if name in existing_names:
                    log_message(f"Entry already registered for {name}.")
                else:
                    timestamp = datetime.now().strftime('%H:%M:%S') 
                    date_col_entry = f"Entry Time_{today.strftime('%Y-%m-%d')}"
                    data.append({
                        "S.No": person_count,
                        "Identified Person": name,
                        date_col_entry: timestamp
                    })
                    person_count += 1
                    log_message(f"New entry registered for {name}.")

    new_df = pd.DataFrame(data)

    if not new_df.empty:
        if existing_df.empty:
            final_df = new_df
        else:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = existing_df

    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
        final_df.to_excel(writer, index=False)

    manage_old_files()

def on_exit_button_click():
    save_directory = "t"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    def calculate_brightness(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image)

    brightness_threshold = 100

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log_message("Error: Could not open webcam.")
        return

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            log_message("Error: Could not read frame.")
            break

        brightness = calculate_brightness(frame)
        face_locations = face_recognition.face_locations(frame)

        if face_locations and brightness > brightness_threshold:
            image_filename = os.path.join(save_directory, "captured_image.jpg")
            cv2.imwrite(image_filename, frame)
            log_message(f"Face detected. Image saved at {image_filename}")
            break

        if time.time() - start_time > 10:
            log_message("Exiting loop due to timeout.")
            break

    cap.release()

    folder_path = "t"
    known_faces_path = "known_faces"
    data = []

    known_faces = []
    known_names = []

    for filename in os.listdir(known_faces_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    today = datetime.now().date()
    excel_file = get_excel_filename()

    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        existing_names = existing_df["Identified Person"].tolist()
        if f"Exit Time_{today.strftime('%Y-%m-%d')}" not in existing_df.columns:
            existing_df[f"Exit Time_{today.strftime('%Y-%m-%d')}"] = None

    else:
        existing_df = pd.DataFrame(columns=["Identified Person"])
        existing_df[f"Exit Time_{today.strftime('%Y-%m-%d')}"] = None
        existing_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                if name in existing_names:
                    index = existing_df[existing_df["Identified Person"] == name].index[0]
                    if pd.isna(existing_df.at[index, f"Exit Time_{today.strftime('%Y-%m-%d')}"]):
                        existing_df.at[index, f"Exit Time_{today.strftime('%Y-%m-%d')}"] = datetime.now().strftime('%H:%M:%S')
                        log_message(f"Exit time registered for {name}.")
                    else:
                        log_message(f"Exit time already registered for {name}.")
                else:
                    log_message(f"Entry not registered for {name}.")

    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
        existing_df.to_excel(writer, index=False)

    wb = load_workbook(excel_file)
    ws = wb.active

    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter  
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        adjusted_width = (max_length + 2)
        ws.column_dimensions[column].width = adjusted_width

    wb.save(excel_file)

    manage_old_files()

root = tk.Tk()
root.title("Simple GUI")

log_widget = scrolledtext.ScrolledText(root, width=60, height=20)
log_widget.pack(pady=10)

entry_button = tk.Button(root, text="Entry", command=on_entry_button_click)
entry_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=on_exit_button_click)
exit_button.pack(pady=10)

root.mainloop()

# pip install opencv-python face-recognition pandas openpyxl
