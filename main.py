import tkinter as tk
from tkinter import ttk, messagebox
import employee_registration
import model_training
import attendance_recognition
import attendance_viewer
import utils
import logging

class SmartAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Setup logging
        logging.basicConfig(
            filename='error.log',
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#34495e', height=100)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="Smart Attendance System",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#34495e'
        )
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(expand=True)
        
        # Buttons
        buttons = [
            ("Employee Registration", self.open_registration, '#3498db'),
            ("Train Models", self.open_training, '#2ecc71'),
            ("Mark Attendance", self.open_attendance, '#e74c3c'),
            ("View Attendance", self.open_viewer, '#f39c12'),
            ("Exit", self.exit_app, '#95a5a6')
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(
                button_frame,
                text=text,
                command=command,
                font=('Arial', 14),
                bg=color,
                fg='white',
                width=20,
                height=2,
                relief='flat',
                bd=0
            )
            btn.pack(pady=10)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg='#34495e'))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.configure(bg=c))
    
    def open_registration(self):
        try:
            employee_registration.EmployeeRegistrationWindow(self.root)
        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            messagebox.showerror("Error", "Failed to open registration window")
    
    def open_training(self):
        try:
            model_training.ModelTrainingWindow(self.root)
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            messagebox.showerror("Error", "Failed to open training window")
    
    def open_attendance(self):
        try:
            attendance_recognition.AttendanceRecognitionWindow(self.root)
        except Exception as e:
            logging.error(f"Attendance error: {str(e)}")
            messagebox.showerror("Error", "Failed to open attendance window")
    
    def open_viewer(self):
        try:
            attendance_viewer.AttendanceViewerWindow(self.root)
        except Exception as e:
            logging.error(f"Viewer error: {str(e)}")
            messagebox.showerror("Error", "Failed to open attendance viewer")
    
    def exit_app(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartAttendanceSystem(root)
    root.mainloop()
