import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
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

    
root = tk.Tk()
root.title("FaceSecure")
root.geometry("900x500")
root.configure(bg="#76b5c5")

logo_path = "img3.jpg" 
logo_img = Image.open(logo_path)
logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
logo_tk = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(root, image=logo_tk, bg="#76b5c5")
logo_label.pack(pady=(10, 5))

title_label = tk.Label(root, text="FaceSecure", font=("Arial", 28, "bold"), fg="#5E1AB8", bg="#76b5c5")
title_label.pack()

button_frame = tk.Frame(root, bg="#76b5c5")
button_frame.pack(pady=10)

btn_style = {"font": ("Arial", 12, "bold"), "fg": "white", "bg": "#5E1AB8", "width": 18, "height": 2}

review_leave_btn = tk.Button(button_frame, text="Review Leave", command=review_leave_requests, **btn_style)
review_leave_btn.grid(row=0, column=0, padx=15)

subject_var = tk.StringVar()
subjects = get_subjects()

subject_dropdown = ttk.Combobox(button_frame, textvariable=subject_var, values=subjects, state='readonly', font=("Arial", 12))
subject_dropdown.grid(row=0, column=1, padx=15)
subject_dropdown.current(0)

send_warnings_btn = tk.Button(button_frame, text="Show Attendance", command=load_attendance, **btn_style)
send_warnings_btn.grid(row=0, column=2, padx=15)

send_warnings_btn = tk.Button(button_frame, text="Send Warnings", command=send_warnings, **btn_style)
send_warnings_btn.grid(row=0, column=3, padx=15)

table_frame = tk.Frame(root, bg="white")
table_frame.pack(pady=10, fill="both", expand=True)

columns = ("Roll Number", "Name", "Attendance %")
tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
for col in columns:
    tree.heading(col, text=col, anchor="center")
    tree.column(col, width=200, anchor="center")

tree.pack(fill="both", expand=True)

root.mainloop()