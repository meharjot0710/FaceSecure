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
    leave_request = pending_requests.iloc[0]
    roll_number = leave_request["Roll Number"]
    name = leave_request["Name"]
    start_date = leave_request["Start Date"]
    end_date = leave_request["End Date"]
    result = messagebox.askquestion("Approve Leave", f"Do you want to approve the leave for {name} (Roll No: {roll_number}) from {start_date} to {end_date}?")
    if result == 'yes':
        df.loc[df["Roll Number"] == roll_number, "Status"] = "Accepted"
        df.to_excel(leave_file, index=False)
        update_attendance_criteria(roll_number)
        messagebox.showinfo("Leave Accepted", f"Leave for {name} (Roll No: {roll_number}) has been accepted.")
    else:
        df.loc[df["Roll Number"] == roll_number, "Status"] = "Rejected"
        df.to_excel(leave_file, index=False)
        messagebox.showinfo("Leave Rejected", f"Leave for {name} (Roll No: {roll_number}) has been rejected.")

root = tk.Tk()
root.title("FaceSecure")

button_frame = tk.Frame(root)
button_frame.pack()

subject_var = tk.StringVar()
subjects = get_subjects()

ttk.Label(root, text="Select Subject:").pack()
subject_dropdown = ttk.Combobox(root, textvariable=subject_var, values=subjects, state='readonly')
subject_dropdown.pack()
subject_dropdown.current(0)

ttk.Button(root, text="View Attendance", command=load_attendance).pack()
ttk.Button(root, text="Send Warnings", command=send_warnings).pack()
ttk.Button(button_frame, text="Review Leave Requests", command=review_leave_requests).grid(row=2, column=1, padx=15, pady=10)


columns = ("Roll Number", "Name", "Attendance %")
tree = ttk.Treeview(root, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=120)
tree.pack()
root.mainloop()

# def setup_gui():
#     global log_widget
#     root = tk.Tk()
#     root.title("FaceSecure")
#     root.geometry("700x500")
#     root.configure(bg="#2C3E50")
#     style = ttk.Style()
#     style.configure("TButton", font=("Arial", 12), padding=10)
#     style.configure("T`Label", font=("Arial", 14), background="#2C3E50", foreground="white")
#     header_frame = tk.Frame(root, bg="#1A252F", pady=10)
#     header_frame.pack(fill=tk.X)
#     title_label = tk.Label(header_frame, text="Attendance", font=("Arial", 18, "bold"), bg="#1A252F", fg="white")
#     title_label.pack()
#     main_frame = tk.Frame(root, bg="#2C3E50", padx=20, pady=20)
#     main_frame.pack(expand=True)
#     button_frame = tk.Frame(main_frame, bg="#2C3E50")
#     button_frame.pack(pady=10)
#     ttk.Button(button_frame, text="Mark Attendance", command=process_image).grid(row=0, column=0, padx=15, pady=10)
#     ttk.Button(button_frame, text="Add Face", command=add_face).grid(row=0, column=1, padx=15, pady=10)
#     ttk.Button(button_frame, text="Check Warnings", command=check_warnings).grid(row=1, column=0, padx=15, pady=10)
#     ttk.Button(button_frame, text="Attendance Prediction", command=predict_attendance).grid(row=1, column=1, padx=15, pady=10)
#     log_label = ttk.Label(main_frame, text="Logs:")
#     log_label.pack(anchor="w", pady=(20, 5))
#     log_widget = tk.Text(main_frame, width=80, height=15, wrap="word", bg="#ECF0F1", fg="#2C3E50", font=("Arial", 10))
#     log_widget.pack()
#     root.mainloop()

# if __name__ == "__main__":
#     initialize_excel()
#     setup_gui()