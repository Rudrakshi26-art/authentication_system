from tkinter import Tk
from face_recognition import FaceRecognition
from student_management import StudentManagement
from attendance_system import AttendanceSystem
from gesture import GestureRecognition
from ui_components import UIComponents

class FaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.students = []  # Shared student data
        self.student_id = [0]  # Shared ID counter as list for mutability

        # Initialize modules
        self.face_recognizer = FaceRecognition()
        self.student_manager = StudentManagement(self.students, self.student_id)
        self.attendance_system = AttendanceSystem(self.students)
        self.gesture_recognizer = GestureRecognition()

        # Initialize UI with all modules
        self.ui = UIComponents(root, self.face_recognizer, self.student_manager, self.attendance_system, self.gesture_recognizer)

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionSystem(root)
    root.mainloop()
