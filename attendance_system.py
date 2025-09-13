from tkinter import Toplevel, Label, Listbox, Button, messagebox
import datetime
import csv
import os

class AttendanceSystem:
    def __init__(self, students):
        self.students = students

    def manual_attendance_window(self, root):
        """Open manual attendance marking window"""
        if not self.students:
            messagebox.showerror("Error", "No students added yet. Please add students first.")
            return

        self.att_window = Toplevel(root)
        self.att_window.title("Attendance System")
        self.att_window.geometry("500x400")

        # Label
        label = Label(self.att_window, text="Select Student to Mark Attendance:")
        label.pack(pady=10)

        # Listbox for students
        self.student_listbox = Listbox(self.att_window, width=50)
        self.student_listbox.pack(pady=10)

        # Populate listbox
        for student in self.students:
            self.student_listbox.insert("end", f"{student['name']} - Roll: {student['roll']} - Dept: {student['dept']}")

        # Button to mark attendance
        mark_btn = Button(self.att_window, text="Mark Present", command=self.mark_manual_attendance)
        mark_btn.pack(pady=10)

    def mark_manual_attendance(self):
        """Mark attendance for selected student"""
        selected = self.student_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a student!")
            return

        index = selected[0]
        student = self.students[index]

        # Save attendance to file
        self.save_attendance_to_file(student)

        messagebox.showinfo("Attendance Marked", f"Attendance marked for {student['name']} (Roll: {student['roll']})")
        self.att_window.destroy()

    def save_attendance_to_file(self, student):
        """Save attendance to CSV file"""
        attendance_file = "attendance.csv"
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Check if file exists, if not create with headers
        file_exists = os.path.exists(attendance_file)

        with open(attendance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Date", "Time", "Student_ID", "Name", "Roll_Number", "Department"])

            writer.writerow([date_str, time_str, student['id'], student['name'], student['roll'], student['dept']])

    def get_student_by_id(self, student_id):
        """Get student by ID for attendance marking"""
        for student in self.students:
            if student['id'] == student_id:
                return student
        return None

    def mark_automatic_attendance(self, student_id, confidence):
        """Mark attendance automatically when face is recognized"""
        student = self.get_student_by_id(student_id)
        if student:
            # Save attendance to file
            self.save_attendance_to_file(student)
            return f"Attendance marked for {student['name']} (Confidence: {confidence:.2f})"
        return None
