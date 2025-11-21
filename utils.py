import os
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime  # Make sure this import exists
import logging
from tkinter import messagebox

class Utils:
    @staticmethod
    def setup_directories():
        """Create necessary directories"""
        directories = ['data', 'models']
        for dir_name in directories:
            os.makedirs(dir_name, exist_ok=True)
    
    @staticmethod
    def get_employee_path(employee_id):
        """Get employee data directory path"""
        return f"data/{employee_id}"
    
    @staticmethod
    def get_face_path(employee_id):
        """Get employee face images directory path"""
        return f"data/{employee_id}/faces"
    
    @staticmethod
    def get_gesture_path(employee_id):
        """Get employee gesture data directory path"""
        return f"data/{employee_id}/gestures"
    
    @staticmethod
    def create_employee_directories(employee_id):
        """Create directories for employee data"""
        paths = [
            Utils.get_employee_path(employee_id),
            Utils.get_face_path(employee_id),
            Utils.get_gesture_path(employee_id)
        ]
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def load_face_cascade():
        """Load face detection cascade"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                raise Exception("Failed to load face cascade")
            return face_cascade
        except Exception as e:
            logging.error(f"Face cascade loading error: {str(e)}")
            messagebox.showerror("Error", "Failed to load face detection model")
            return None
    
    @staticmethod
    def preprocess_face(image):
        """Preprocess face image for training"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    @staticmethod
    def save_attendance(employee_id, name, department):
        """Save attendance record to CSV - FIXED VERSION"""
        try:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M:%S")
            
            # Create record
            record = {
                'employee_id': employee_id,
                'name': name,
                'department': department,
                'date': date_str,
                'time': time_str,
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create DataFrame for new record
            new_df = pd.DataFrame([record])
            
            # Check if attendance file exists
            if os.path.exists('attendance.csv'):
                existing_df = pd.read_csv('attendance.csv')
                
                # Check for duplicate entry for same day - FIXED
                mask = (existing_df['employee_id'] == employee_id) & (existing_df['date'] == date_str)
                if not existing_df[mask].empty:
                    return False, "Attendance already marked for today"
                
                # Combine existing and new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv('attendance.csv', index=False)
            else:
                # Create new file
                new_df.to_csv('attendance.csv', index=False)
            
            return True, "Attendance marked successfully"
            
        except Exception as e:
            logging.error(f"Attendance saving error: {str(e)}")
            return False, f"Error saving attendance: {str(e)}"
    
    @staticmethod
    def load_attendance_records():
        """Load attendance records from CSV"""
        try:
            if os.path.exists('attendance.csv'):
                return pd.read_csv('attendance.csv')
            else:
                return pd.DataFrame(columns=['employee_id', 'name', 'department', 'date', 'time'])
        except Exception as e:
            logging.error(f"Attendance loading error: {str(e)}")
            return pd.DataFrame(columns=['employee_id', 'name', 'department', 'date', 'time'])
    
    @staticmethod
    def load_employee_info(employee_id):
        """Load specific employee information"""
        try:
            info_file = f"data/{employee_id}/info.pkl"
            if os.path.exists(info_file):
                with open(info_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            logging.error(f"Employee info loading error: {str(e)}")
            return None
    
    @staticmethod
    def check_camera():
        """Check if camera is available"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return ret
        except:
            return False
