from tkinter import Toplevel, Label, Entry, Button, messagebox
import os

class StudentManagement:
    def __init__(self, students_list, student_id_counter):
        self.students = students_list
        self.student_id = student_id_counter

    def add_student_window(self, root):
        """Open window to add new student"""
        self.new_window = Toplevel(root)
        self.new_window.title("Add Student Details")
        self.new_window.geometry("400x400")

        # Name
        name_label = Label(self.new_window, text="Name:")
        name_label.grid(row=0, column=0, padx=10, pady=10)
        self.name_entry = Entry(self.new_window)
        self.name_entry.grid(row=0, column=1, padx=10, pady=10)

        # Roll Number
        roll_label = Label(self.new_window, text="Roll Number:")
        roll_label.grid(row=1, column=0, padx=10, pady=10)
        self.roll_entry = Entry(self.new_window)
        self.roll_entry.grid(row=1, column=1, padx=10, pady=10)

        # Department
        dept_label = Label(self.new_window, text="Department:")
        dept_label.grid(row=2, column=0, padx=10, pady=10)
        self.dept_entry = Entry(self.new_window)
        self.dept_entry.grid(row=2, column=1, padx=10, pady=10)

        # Save button
        save_btn = Button(self.new_window, text="Save Student", command=self.save_student)
        save_btn.grid(row=3, column=0, columnspan=2, pady=20)

    def save_student(self):
        """Save student details"""
        name = self.name_entry.get()
        roll = self.roll_entry.get()
        dept = self.dept_entry.get()

        if not name or not roll or not dept:
            messagebox.showerror("Error", "All fields are required!")
            return

        # Store in list with ID
        self.students.append({"id": self.student_id, "name": name, "roll": roll, "dept": dept})
        self.student_id += 1

        messagebox.showinfo("Success", f"Student {name} (Roll: {roll}, Dept: {dept}) added successfully!")
        self.new_window.destroy()

    def get_student_by_id(self, student_id):
        """Get student details by ID"""
        for student in self.students:
            if student['id'] == student_id:
                return student
        return None

    def get_all_students(self):
        """Get all students"""
        return self.students

    def create_student_directory(self, student_id):
        """Create directory for student face images"""
        data_dir = f"data/{student_id}"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    def validate_student_data(self):
        """Validate that students exist"""
        return len(self.students) > 0
