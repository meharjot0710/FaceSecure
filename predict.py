import pandas as pd
import os
from datetime import datetime
from tkinter import simpledialog

def predict_attendance():
    subject = simpledialog.askstring("Input", f"Select subject: {', '.join(subjects)}")
    if subject not in subjects:
        log_message("Invalid subject selection.")
        return

    excel_file = os.path.join("attendance_records", f"{subject}.xlsx")
    df = pd.read_excel(excel_file)
    
    if "Roll Number" not in df.columns:
        log_message("Invalid Excel format. Missing 'Roll Number' column.")
        return
    
    date_columns = [col for col in df.columns if col not in ["Roll Number", "Name"]]
    total_classes = len(date_columns)
    
    if total_classes == 0:
        log_message("No classes conducted yet for prediction.")
        return
    
    predictions = []
    min_attendance_required = 75  # Minimum percentage required
    
    for _, row in df.iterrows():
        roll_number = row["Roll Number"]
        name = row.get("Name", "Unknown")
        
        total_present = sum(row[date] == "P" for date in date_columns)
        current_percentage = (total_present / total_classes) * 100
        
        # Projected future attendance
        remaining_classes = 50  # Assume 50 more classes
        projected_present = total_present + remaining_classes
        projected_percentage = (projected_present / (total_classes + remaining_classes)) * 100
        
        # Calculate how many more classes can be missed
        max_missed = ((total_present * 100) / min_attendance_required) - total_classes
        max_missed = max(0, int(max_missed))
        
        predictions.append(
            f"{name} (Roll No: {roll_number}): Current: {current_percentage:.2f}%, "
            f"Projected: {projected_percentage:.2f}%, Can miss {max_missed} more classes"
        )
    
    log_message("\n".join(predictions))
