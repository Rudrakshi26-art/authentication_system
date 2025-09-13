from tkinter import Label, Button, Listbox, Toplevel, messagebox
from PIL import Image, ImageTk
import cv2
import os

class UIComponents:
    def __init__(self, root, face_recognizer, student_manager, attendance_system, gesture_recognizer):
        self.root = root
        self.face_recognizer = face_recognizer
        self.student_manager = student_manager
        self.attendance_system = attendance_system
        self.gesture_recognizer = gesture_recognizer
        self.setup_main_ui()

    def setup_main_ui(self):
        """Setup the main user interface"""
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # Load and place background images
        self.load_background_images()

        # Title label
        title_lbl = Label(self.bg_img, text="FACE RECOGNITION ATTENDANCE SYSTEM SOFTWARE",
                          font=("times new roman", 35, "bold"), bg="white", fg="red")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # Create buttons
        self.create_buttons()

    def load_background_images(self):
        """Load and place background images"""
        # First image
        img = Image.open(r"college image/face 14.jpg")
        img = img.resize((500, 130), Image.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)
        f_lbl = Label(self.root, image=self.photoimg)
        f_lbl.place(x=0, y=0, width=500, height=130)

        # Second image
        img1 = Image.open(r"college image/face 15.png")
        img1 = img1.resize((500, 130), Image.LANCZOS)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl1 = Label(self.root, image=self.photoimg1)
        f_lbl1.place(x=500, y=0, width=500, height=130)

        # Third image
        img2 = Image.open(r"college image/face 13.jpg")
        img2 = img2.resize((500, 130), Image.LANCZOS)
        self.photoimg2 = ImageTk.PhotoImage(img2)
        f_lbl2 = Label(self.root, image=self.photoimg2)
        f_lbl2.place(x=1000, y=0, width=550, height=130)

        # Background image
        img3 = Image.open(r"college image/background 11.jpg")
        img3 = img3.resize((1530, 710), Image.LANCZOS)
        self.photoimg3 = ImageTk.PhotoImage(img3)
        self.bg_img = Label(self.root, image=self.photoimg3)
        self.bg_img.place(x=0, y=130, width=1530, height=710)

    def create_buttons(self):
        """Create all the main buttons"""
        # Student button
        img4 = Image.open(r"college image/student 4.jpg")
        img4 = img4.resize((220, 220), Image.LANCZOS)
        self.photoimg4 = ImageTk.PhotoImage(img4)
        b1 = Button(self.bg_img, image=self.photoimg4, cursor="hand2", command=lambda: self.student_manager.add_student_window(self.root))
        b1.place(x=200, y=100, width=220, height=220)
        b1_txt = Label(self.bg_img, text="Student Details", font=("times new roman", 15, "bold"),
                       bg="darkblue", fg="white")
        b1_txt.place(x=200, y=320, width=220, height=40)

        # Face Detection button
        img5 = Image.open(r"college image/face 2.jpg")
        img5 = img5.resize((220, 220), Image.LANCZOS)
        self.photoimg5 = ImageTk.PhotoImage(img5)
        b2 = Button(self.bg_img, image=self.photoimg5, cursor="hand2", command=self.face_recognizer.real_time_face_detection)
        b2.place(x=500, y=100, width=220, height=220)
        b2_txt = Label(self.bg_img, text="Face Detection", font=("times new roman", 15, "bold"),
                       bg="darkblue", fg="white")
        b2_txt.place(x=500, y=320, width=220, height=40)

        # Attendance button
        img6 = Image.open(r"college image/attendence 5.jpg")
        img6 = img6.resize((220, 220), Image.LANCZOS)
        self.photoimg6 = ImageTk.PhotoImage(img6)
        b3 = Button(self.bg_img, image=self.photoimg6, cursor="hand2", command=lambda: self.attendance_system.manual_attendance_window(self.root))
        b3.place(x=800, y=100, width=220, height=220)
        b3_txt = Label(self.bg_img, text="Attendance", font=("times new roman", 15, "bold"),
                       bg="darkblue", fg="white")
        b3_txt.place(x=800, y=320, width=220, height=40)

        # Train Data button
        img7 = Image.open(r"college image/train data 7.jpg")
        img7 = img7.resize((220, 220), Image.LANCZOS)
        self.photoimg7 = ImageTk.PhotoImage(img7)
        b4 = Button(self.bg_img, image=self.photoimg7, cursor="hand2", command=self.train_data_window)
        b4.place(x=1100, y=100, width=220, height=220)
        b4_txt = Label(self.bg_img, text="Train Data", font=("times new roman", 15, "bold"),
                       bg="darkblue", fg="white")
        b4_txt.place(x=1100, y=320, width=220, height=40)

        # Automatic Attendance button
        img8 = Image.open(r"college image/face 2.jpg")  # Using face detection image for now
        img8 = img8.resize((220, 220), Image.LANCZOS)
        self.photoimg8 = ImageTk.PhotoImage(img8)
        b5 = Button(self.bg_img, image=self.photoimg8, cursor="hand2", command=self.start_automatic_attendance)
        b5.place(x=200, y=400, width=220, height=220)
        b5_txt = Label(self.bg_img, text="Auto Attendance", font=("times new roman", 15, "bold"),
                       bg="darkblue", fg="white")
        b5_txt.place(x=200, y=620, width=220, height=40)

        # Gesture Recognition button
        img9 = Image.open(r"college image/student 4.jpg")  # Using student image for now
        img9 = img9.resize((220, 220), Image.LANCZOS)
        self.photoimg9 = ImageTk.PhotoImage(img9)
        b6 = Button(self.bg_img, image=self.photoimg9, cursor="hand2", command=self.start_gesture_recognition)
        b6.place(x=500, y=400, width=220, height=220)
        b6_txt = Label(self.bg_img, text="Gesture Recognition", font=("times new roman", 15, "bold"),
                       bg="darkblue", fg="white")
        b6_txt.place(x=500, y=620, width=220, height=40)

        # Gesture Controlled Attendance button
        img10 = Image.open(r"college image/attendence 5.jpg")  # Using attendance image
        img10 = img10.resize((220, 220), Image.LANCZOS)
        self.photoimg10 = ImageTk.PhotoImage(img10)
        b7 = Button(self.bg_img, image=self.photoimg10, cursor="hand2", command=self.start_gesture_controlled_attendance)
        b7.place(x=800, y=400, width=220, height=220)
        b7_txt = Label(self.bg_img, text="Gesture Control", font=("times new roman", 15, "bold"),
                       bg="darkblue", fg="white")
        b7_txt.place(x=800, y=620, width=220, height=40)



    def train_data_window(self):
        """Open training data window"""
        if not self.student_manager.validate_student_data():
            messagebox.showerror("Error", "No students added. Please add students first.")
            return

        self.train_window = Toplevel(self.root)
        self.train_window.title("Train Face Recognition Model")
        self.train_window.geometry("500x400")

        # Label
        label = Label(self.train_window, text="Select Student to Train:")
        label.pack(pady=10)

        # Listbox for students
        self.train_listbox = Listbox(self.train_window, width=50)
        self.train_listbox.pack(pady=10)

        # Populate listbox
        for student in self.student_manager.get_all_students():
            self.train_listbox.insert("end", f"{student['name']} - Roll: {student['roll']} - Dept: {student['dept']}")

        # Button to start training
        train_btn = Button(self.train_window, text="Start Training", command=self.start_training)
        train_btn.pack(pady=10)

    def start_training(self):
        """Start the training process"""
        selected = self.train_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a student!")
            return

        index = selected[0]
        student = self.student_manager.get_all_students()[index]

        # Create directory for student
        self.student_manager.create_student_directory(student['id'])

        # Capture faces
        count = self.face_recognizer.capture_face_images(student['id'])

        if count == 0:
            messagebox.showerror("Error", "No faces captured")
            return

        # Train model
        if self.face_recognizer.train_model(self.student_manager.get_all_students()):
            messagebox.showinfo("Success", f"Model trained successfully for {student['name']}!")
        else:
            messagebox.showerror("Error", "Training failed")

        self.train_window.destroy()

    def start_automatic_attendance(self):
        """Start the automatic attendance system"""
        self.face_recognizer.real_time_face_recognition_attendance(self.attendance_system)

    def start_gesture_recognition(self):
        """Start gesture recognition"""
        self.gesture_recognizer.real_time_gesture_recognition()

    def start_gesture_controlled_attendance(self):
        """Start gesture-controlled attendance system"""
        self.gesture_recognizer.gesture_controlled_attendance(self.attendance_system, self.face_recognizer)


