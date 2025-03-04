import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

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

root = tk.Tk()
root.title("Teacher Dashboard - View Attendance")

subject_var = tk.StringVar()
subjects = get_subjects()

ttk.Label(root, text="Select Subject:").pack()
subject_dropdown = ttk.Combobox(root, textvariable=subject_var, values=subjects, state='readonly')
subject_dropdown.pack()
subject_dropdown.current(0)

ttk.Button(root, text="View Attendance", command=load_attendance).pack()
ttk.Button(root, text="Send Warnings", command=send_warnings).pack()

columns = ("Roll Number", "Name", "Attendance %")
tree = ttk.Treeview(root, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=120)
tree.pack()
root.mainloop()