import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import mediapipe as mp
import pickle
import numpy as np
from utils import Utils
import logging
import time

class EmployeeRegistrationWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Employee Registration")
        self.window.geometry("900x700")
        self.window.configure(bg='#2c3e50')
        
        self.employee_data = {}
        self.face_images = []
        self.gesture_landmarks = None
        self.cap = None
        self.face_cascade = Utils.load_face_cascade()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_count = 0
        self.max_faces = 20
        self.is_capturing_faces = False
        self.is_capturing_gesture = False
        self.last_capture_time = 0
        self.capture_interval = 0.5
        
        # Gesture capture variables
        self.gesture_countdown_started = False
        self.gesture_capture_start_time = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Form frame
        form_frame = ttk.LabelFrame(main_frame, text="Employee Details", padding=10)
        form_frame.pack(fill='x', pady=(0, 10))
        
        # Form fields
        fields = [
            ("Employee ID*", "employee_id"),
            ("Full Name*", "name"),
            ("Department*", "department"),
            ("Email", "email"),
            ("Phone", "phone")
        ]
        
        self.entries = {}
        for i, (label, key) in enumerate(fields):
            ttk.Label(form_frame, text=label).grid(row=i, column=0, sticky='w', pady=5)
            entry = ttk.Entry(form_frame, width=30)
            entry.grid(row=i, column=1, sticky='ew', pady=5, padx=(10, 0))
            self.entries[key] = entry
        
        form_frame.columnconfigure(1, weight=1)
        
        # Camera frame
        camera_frame = ttk.LabelFrame(main_frame, text="Face & Gesture Capture", padding=10)
        camera_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Video display
        self.video_label = ttk.Label(camera_frame)
        self.video_label.pack(pady=10)
        
        # Instructions
        self.instructions_var = tk.StringVar(value="Fill the form and click 'Start Face Capture'")
        instructions_label = ttk.Label(camera_frame, textvariable=self.instructions_var, 
                                     font=('Arial', 10, 'bold'), foreground='blue')
        instructions_label.pack(pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(camera_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(camera_frame)
        button_frame.pack(pady=10)
        
        self.face_capture_btn = ttk.Button(button_frame, text="Start Face Capture", 
                                         command=self.toggle_face_capture)
        self.face_capture_btn.pack(side='left', padx=5)
        
        self.gesture_capture_btn = ttk.Button(button_frame, text="Capture Gesture", 
                                            command=self.start_gesture_capture)
        self.gesture_capture_btn.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Save Employee", 
                  command=self.save_employee).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", 
                  command=self.close).pack(side='left', padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(camera_frame, mode='determinate', maximum=self.max_faces)
        self.progress.pack(fill='x', pady=5)
        
        self.progress_label = ttk.Label(camera_frame, text=f"Faces captured: 0/{self.max_faces}")
        self.progress_label.pack()
        
        # Bind window close event
        self.window.protocol("WM_DELETE_WINDOW", self.close)
    
    def toggle_face_capture(self):
        """Toggle face capture on/off"""
        if not self.validate_form():
            return
            
        if not self.is_capturing_faces:
            self.start_face_capture()
        else:
            self.stop_face_capture()
    
    def start_face_capture(self):
        """Start the face capture process"""
        if not self.validate_form():
            return
            
        # Stop gesture capture if running
        if self.is_capturing_gesture:
            self.stop_gesture_capture()
            
        self.face_count = 0
        self.face_images = []
        self.is_capturing_faces = True
        self.face_capture_btn.configure(text="Stop Face Capture")
        self.instructions_var.set("Position your face in the camera. Face capture will auto-start.")
        self.status_var.set("Starting face capture...")
        
        if not self.start_camera():
            self.is_capturing_faces = False
            self.face_capture_btn.configure(text="Start Face Capture")
            return
        
        self.update_camera_feed()
    
    def stop_face_capture(self):
        """Stop the face capture process"""
        self.is_capturing_faces = False
        self.face_capture_btn.configure(text="Start Face Capture")
        self.instructions_var.set("Face capture stopped")
        self.status_var.set(f"Captured {self.face_count} faces")
    
    def start_gesture_capture(self):
        """Start gesture capture process"""
        if not self.validate_form():
            return
        
        # Stop face capture if running
        if self.is_capturing_faces:
            self.stop_face_capture()
            
        self.is_capturing_gesture = True
        self.gesture_capture_btn.configure(text="Stop Gesture Capture")
        self.instructions_var.set("Show your hand gesture (✋) in the camera")
        self.status_var.set("Waiting for hand gesture...")
        
        if not self.start_camera():
            self.is_capturing_gesture = False
            self.gesture_capture_btn.configure(text="Capture Gesture")
            return
        
        # Reset gesture capture variables
        self.gesture_landmarks = None
        self.gesture_countdown_started = False
        self.gesture_capture_start_time = 0
        
        self.update_gesture_feed()
    
    def stop_gesture_capture(self):
        """Stop gesture capture process"""
        self.is_capturing_gesture = False
        self.gesture_countdown_started = False
        self.gesture_capture_btn.configure(text="Capture Gesture")
        self.instructions_var.set("Gesture capture stopped")
    
    def validate_form(self):
        """Validate required form fields"""
        required_fields = ['employee_id', 'name', 'department']
        for field in required_fields:
            if not self.entries[field].get().strip():
                messagebox.showerror("Error", f"Please fill in {field.replace('_', ' ').title()}")
                return False
        return True
    
    def start_camera(self):
        """Initialize camera"""
        try:
            # Try different camera indices if 0 fails
            for i in range(5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    logging.info(f"Camera opened successfully with index {i}")
                    break
                self.cap.release()

            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera. Please check if another application is using the camera or if camera permissions are enabled.")
                return False

            # Set camera resolution for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            return True
        except Exception as e:
            logging.error(f"Camera error: {str(e)}")
            messagebox.showerror("Error", f"Camera initialization failed: {str(e)}")
            return False
    
    def update_camera_feed(self):
        """Update camera feed for face capture"""
        if self.cap is None or not self.is_capturing_faces:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            face_detected = len(faces) > 0
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add face detection text
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Auto-capture face if conditions are met
                current_time = time.time()
                if (self.is_capturing_faces and 
                    self.face_count < self.max_faces and 
                    len(faces) == 1 and
                    current_time - self.last_capture_time > self.capture_interval):
                    
                    # Additional check: face should be reasonably centered and sized
                    if self.is_good_face_position(frame, x, y, w, h):
                        self.capture_face(frame, x, y, w, h)
                        self.last_capture_time = current_time
            
            # Add guidance text
            if self.face_count < self.max_faces:
                if face_detected:
                    status_text = f"Face detected! Capturing... ({self.face_count}/{self.max_faces})"
                    color = (0, 255, 0)
                else:
                    status_text = "Please position your face in the camera"
                    color = (0, 0, 255)
                
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "Face capture completed!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.stop_face_capture()
            
            # Display frame
            self.display_frame(frame)
            
            if self.is_capturing_faces and self.face_count < self.max_faces:
                self.window.after(10, self.update_camera_feed)
            else:
                self.stop_face_capture()
    
    def update_gesture_feed(self):
        """Update camera feed for gesture capture - Countdown only when hand is visible"""
        if self.cap is None or not self.is_capturing_gesture:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_detected = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    self.gesture_landmarks = landmarks
                    hand_detected = True
                    
                    # Start countdown only when hand is first detected
                    if not self.gesture_countdown_started:
                        self.gesture_countdown_started = True
                        self.gesture_capture_start_time = time.time()
                        self.status_var.set("Hand detected! Starting countdown...")
                    
                    # Check if countdown is complete (3 seconds after hand detection)
                    if self.gesture_countdown_started:
                        current_time = time.time()
                        time_elapsed = current_time - self.gesture_capture_start_time
                        remaining = max(0, 3.0 - time_elapsed)
                        
                        if remaining <= 0:
                            self.auto_save_gesture()
                            return
            else:
                # Reset countdown if hand is lost
                if self.gesture_countdown_started:
                    self.gesture_countdown_started = False
                    self.status_var.set("Hand lost! Show your gesture again")
            
            # Add instruction text
            if hand_detected and self.gesture_countdown_started:
                current_time = time.time()
                time_elapsed = current_time - self.gesture_capture_start_time
                remaining = max(0, 3.0 - time_elapsed)
                status_text = f"Keep hand steady! Capturing in {remaining:.1f} seconds..."
                color = (0, 255, 0)
                
                # Draw countdown circle
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                radius = 50
                progress = time_elapsed / 3.0  # 0 to 1 over 3 seconds
                
                # Draw background circle
                cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), -1)
                
                # Draw progress arc
                end_angle = int(360 * progress)
                cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 0, end_angle, (0, 255, 0), 10)
                
                # Draw countdown text
                cv2.putText(frame, f"{remaining:.1f}s", (center_x - 20, center_y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            elif hand_detected:
                status_text = "Hand detected! Get ready for countdown..."
                color = (0, 255, 255)
            else:
                status_text = "Show your hand gesture (✋) in the camera"
                color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            self.display_frame(frame)
            
            # Continue updating if still capturing
            if self.is_capturing_gesture:
                self.window.after(10, self.update_gesture_feed)
    
    def auto_save_gesture(self):
        """Automatically save gesture when countdown completes"""
        if self.gesture_landmarks is not None:
            self.status_var.set("Gesture captured successfully!")
            self.instructions_var.set("You can now save the employee")
            self.stop_gesture_capture()
            self.stop_camera()
            
            # Visual feedback with success message
            messagebox.showinfo("Success", "Gesture captured successfully!")
        else:
            self.status_var.set("Gesture capture failed. Please try again.")
            self.gesture_countdown_started = False
    
    def is_good_face_position(self, frame, x, y, w, h):
        """Check if face is in good position for capture"""
        height, width = frame.shape[:2]
        
        # Check if face is reasonably centered
        center_x = x + w/2
        center_y = y + h/2
        
        # Face should be within central 70% of frame
        margin_x = width * 0.15
        margin_y = height * 0.15
        
        good_position = (margin_x < center_x < width - margin_x and 
                        margin_y < center_y < height - margin_y)
        
        # Face should be reasonably sized (not too small)
        good_size = w > 100 and h > 100
        
        return good_position and good_size
    
    def capture_face(self, frame, x, y, w, h):
        """Capture and save face image"""
        try:
            # Extract face region with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_roi = frame[y1:y2, x1:x2]
            
            # Resize to standard size for consistency
            face_roi = cv2.resize(face_roi, (200, 200))
            
            self.face_images.append(face_roi)
            self.face_count += 1
            
            # Update UI
            self.progress['value'] = self.face_count
            self.progress_label.configure(text=f"Faces captured: {self.face_count}/{self.max_faces}")
            self.status_var.set(f"Captured face {self.face_count}/{self.max_faces}")
            
            # Visual feedback
            self.show_capture_feedback(frame)
            
        except Exception as e:
            logging.error(f"Face capture error: {str(e)}")
    
    def show_capture_feedback(self, frame):
        """Show visual feedback when face is captured"""
        # Flash the screen green briefly
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        cv2.putText(frame, "CAPTURED!", (frame.shape[1]//2 - 60, frame.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def display_frame(self, frame):
        """Display frame in the GUI"""
        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        height, width = rgb_frame.shape[:2]
        ratio = min(500/width, 400/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Convert to PhotoImage
        img = tk.PhotoImage(data=cv2.imencode('.png', resized_frame)[1].tobytes())
        self.video_label.configure(image=img)
        self.video_label.image = img
    
    def save_employee(self):
        """Save employee data, faces, and gesture"""
        if not self.validate_form():
            return
            
        if len(self.face_images) < 5:
            messagebox.showwarning("Warning", f"Please capture at least 5 face images. Currently: {len(self.face_images)}")
            return
            
        if self.gesture_landmarks is None:
            messagebox.showwarning("Warning", "Please capture a gesture sample")
            return
            
        employee_id = self.entries['employee_id'].get().strip()
        
        try:
            # Create directories
            Utils.create_employee_directories(employee_id)
            
            # Save face images
            face_path = Utils.get_face_path(employee_id)
            for i, face_img in enumerate(self.face_images):
                cv2.imwrite(f"{face_path}/face_{i:03d}.jpg", face_img)
            
            # Save gesture data
            gesture_path = Utils.get_gesture_path(employee_id)
            with open(f"{gesture_path}/gesture.pkl", 'wb') as f:
                pickle.dump({
                    'landmarks': self.gesture_landmarks,
                    'employee_id': employee_id,
                    'timestamp': time.time()
                }, f)
            
            # Save employee info
            employee_info = {
                key: entry.get().strip() for key, entry in self.entries.items()
            }
            employee_info['registration_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(f"data/{employee_id}/info.pkl", 'wb') as f:
                pickle.dump(employee_info, f)
            
            messagebox.showinfo("Success", 
                              f"Employee registered successfully!\n"
                              f"Face images: {len(self.face_images)}\n"
                              f"Employee ID: {employee_id}")
            self.close()
            
        except Exception as e:
            logging.error(f"Employee save error: {str(e)}")
            messagebox.showerror("Error", f"Failed to save employee: {str(e)}")
    
    def stop_camera(self):
        """Stop camera and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def close(self):
        """Close the window and cleanup"""
        self.stop_camera()
        self.is_capturing_faces = False
        self.is_capturing_gesture = False
        self.gesture_countdown_started = False
        
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
            
        self.window.destroy()
