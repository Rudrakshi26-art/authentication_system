import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import pickle
import numpy as np
import mediapipe as mp
from utils import Utils
import logging
import os
import time
import pandas as pd
from datetime import datetime

class AttendanceRecognitionWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Attendance Recognition")
        self.window.geometry("800x600")
        self.window.configure(bg='#2c3e50')
        
        self.cap = None
        self.face_recognizer = None
        self.gesture_model = None
        self.face_cascade = Utils.load_face_cascade()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.employee_info = {}
        self.attendance_marked_today = set()
        
        self.load_todays_attendance()
        self.load_models()
        self.load_all_employee_info()
        self.setup_ui()
        self.start_camera()
    
    def load_todays_attendance(self):
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            if os.path.exists('attendance.csv'):
                df = pd.read_csv('attendance.csv')
                today_records = df[df['date'] == today]
                self.attendance_marked_today = set(today_records['employee_id'].astype(str))
        except Exception as e:
            logging.error(f"Error loading today's attendance: {str(e)}")
            self.attendance_marked_today = set()
    
    def load_models(self):
        try:
            if os.path.exists('models/face_trainer.yml'):
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.face_recognizer.read('models/face_trainer.yml')
            if os.path.exists('models/gesture_data.pkl'):
                with open('models/gesture_data.pkl', 'rb') as f:
                    self.gesture_model = pickle.load(f)
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def load_all_employee_info(self):
        try:
            if os.path.exists('data'):
                for emp_dir in os.listdir('data'):
                    info_file = os.path.join('data', emp_dir, 'info.pkl')
                    if os.path.exists(info_file):
                        with open(info_file, 'rb') as f:
                            self.employee_info[emp_dir] = pickle.load(f)
        except Exception as e:
            logging.error(f"Employee info loading error: {str(e)}")
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        video_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding=10)
        video_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True)
        
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Ready for recognition")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=('Arial', 12, 'bold'))
        status_label.pack()
        
        info_frame = ttk.LabelFrame(main_frame, text="Recognition Info", padding=10)
        info_frame.pack(fill='x', pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=6, width=50)
        self.info_text.pack(fill='x')
        
        debug_frame = ttk.LabelFrame(main_frame, text="Debug Info", padding=10)
        debug_frame.pack(fill='x', pady=(0, 10))
        
        self.debug_text = tk.Text(debug_frame, height=3, width=50, bg='lightgray')
        self.debug_text.pack(fill='x')
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(button_frame, text="Stop", command=self.stop).pack(side='right', padx=5)
    
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.update_camera_feed()
        except Exception as e:
            messagebox.showerror("Error", "Camera initialization failed")
    
    def update_camera_feed(self):
        if self.cap is None: return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            face_id, face_conf, face_rect = self.detect_face(frame)
            gesture_id, gesture_conf = self.detect_gesture(frame)
            self.process_recognition(frame, face_id, face_conf, gesture_id, gesture_conf, face_rect)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb_frame.shape[:2]
            ratio = min(600/width, 400/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height))
            
            img = tk.PhotoImage(data=cv2.imencode('.png', resized_frame)[1].tobytes())
            self.video_label.configure(image=img)
            self.video_label.image = img
            
            self.window.after(10, self.update_camera_feed)
    
    def detect_face(self, frame):
        if self.face_recognizer is None: return None, 0, None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                label, confidence = self.face_recognizer.predict(face_roi)
                confidence_percentage = max(0, 100 - confidence)
                self.update_debug_info(f"Face - Label: {label}, Raw Conf: {confidence:.2f}, Percentage: {confidence_percentage:.2f}%")
                
                if confidence < 100:
                    return str(label), confidence_percentage, (x, y, w, h)
            except Exception as e:
                logging.error(f"Face prediction error: {str(e)}")
                
        return None, 0, None
    
    def detect_gesture(self, frame):
        if self.gesture_model is None: return None, 0
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                try:
                    prediction = self.gesture_model['model'].predict_proba([landmarks])
                    max_prob = np.max(prediction)
                    predicted_id = self.gesture_model['model'].predict([landmarks])[0]
                    confidence_percentage = max_prob * 100
                    
                    self.update_debug_info(f"Gesture - Label: {predicted_id}, Confidence: {confidence_percentage:.2f}%")
                    
                    # ✅ FIXED: Changed from > to >=
                    if max_prob >= 0.5:  # Now includes 50% exactly
                        return str(predicted_id), confidence_percentage
                except Exception as e:
                    logging.error(f"Gesture prediction error: {str(e)}")
                    
        return None, 0
    
    def update_debug_info(self, message):
        self.debug_text.delete(1.0, 'end')
        self.debug_text.insert(1.0, message)
    
    def process_recognition(self, frame, face_id, face_conf, gesture_id, gesture_conf, face_rect):
        self.info_text.delete(1.0, 'end')
        info_text = ""
        attendance_marked = False
        
        if face_id:
            face_info = self.employee_info.get(face_id, {})
            info_text += f"Face: {face_info.get('name', 'Unknown')} (ID: {face_id}, Conf: {face_conf:.2f}%)\n"
        else:
            info_text += "Face: Not detected\n"
        
        if gesture_id:
            gesture_info = self.employee_info.get(gesture_id, {})
            info_text += f"Gesture: {gesture_info.get('name', 'Unknown')} (ID: {gesture_id}, Conf: {gesture_conf:.2f}%)\n"
        else:
            info_text += "Gesture: Not detected\n"
        
        # Check if both match and confidence thresholds are met
        if face_id and gesture_id and face_id == gesture_id:
            # ✅ FIXED: Changed from 70 to 50 to match gesture detection
            if face_conf >= 50 and gesture_conf >= 50:
                info_text += "\n✅ MATCH CONFIRMED\n"
                
                if face_id in self.attendance_marked_today:
                    info_text += "Attendance: Already marked today\n"
                    self.status_var.set("✅ Attendance Already Marked Today")
                else:
                    employee_info = self.employee_info.get(face_id, {})
                    success, message = Utils.save_attendance(
                        face_id, 
                        employee_info.get('name', 'Unknown'),
                        employee_info.get('department', 'Unknown')
                    )
                    
                    if success:
                        info_text += f"Attendance: {message}\n"
                        self.status_var.set("✅ Attendance Marked Successfully!")
                        self.attendance_marked_today.add(face_id)
                        attendance_marked = True
                        cv2.putText(frame, "ATTENDANCE MARKED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        info_text += f"Attendance Error: {message}\n"
                        self.status_var.set("❌ Attendance Error")
            else:
                info_text += f"\n⚠️ Low confidence (Face: {face_conf:.1f}%, Gesture: {gesture_conf:.1f}%)\n"
                self.status_var.set("⚠️ Low confidence - Please try again")
        elif face_id or gesture_id:
            info_text += "\n❌ Face and gesture don't match\n"
            self.status_var.set("❌ Recognition mismatch")
        else:
            self.status_var.set("🔍 Detecting...")
        
        self.info_text.insert(1.0, info_text)
        
        if attendance_marked:
            self.window.after(3000, self.reset_recognition_status)
    
    def reset_recognition_status(self):
        self.status_var.set("Ready for next recognition")
    
    def stop(self):
        if self.cap: self.cap.release()
        if self.hands: self.hands.close()
        self.window.destroy()
    
    def __del__(self):
        self.stop()
