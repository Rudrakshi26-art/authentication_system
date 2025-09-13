import cv2
import os
import numpy as np
import datetime
from tkinter import messagebox

class FaceRecognition:
    def __init__(self):
        self.recognizer = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.initialize_recognizer()

    def initialize_recognizer(self):
        """Initialize LBPH face recognizer"""
        # Initialize LBPH recognizer
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            from tkinter import messagebox
            messagebox.showerror("Error", "OpenCV 'face' module not found. Please install opencv-contrib-python:\n\npip install opencv-contrib-python")
            self.recognizer = None

        # Load trained model if exists
        if os.path.exists("trainer.yml") and self.recognizer is not None:
            self.recognizer.read("trainer.yml")

    def detect_faces(self, frame):
        """Detect faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces, gray

    def capture_face_images(self, student_id, max_images=100):
        """Capture face images for training"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return 0

        count = 0
        messagebox.showinfo("Capturing", "Capturing face images. Look at the camera. Press 'q' to stop early.")

        while count < max_images:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                cv2.imwrite(f"data/{student_id}/{count}.jpg", face)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('Capturing Faces', frame)
                cv2.waitKey(100)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return count

    def train_model(self, students):
        """Train the face recognition model"""
        faces = []
        labels = []

        # Collect training data
        for student in students:
            sdir = f"data/{student['id']}"
            if os.path.exists(sdir):
                for img_file in os.listdir(sdir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = f"{sdir}/{img_file}"
                        face_img = cv2.imread(img_path, 0)
                        if face_img is not None and face_img.size > 0:
                            faces.append(face_img)
                            labels.append(student['id'])
                        else:
                            print(f"Warning: Could not load image {img_path}")

        # Validate training data
        if not faces:
            messagebox.showerror("Training Error", "No training images found. Please capture face images first.")
            return False

        if len(faces) < 5:
            messagebox.showerror("Training Error", f"Insufficient training data. Found {len(faces)} images, need at least 5.")
            return False

        if self.recognizer is None:
            messagebox.showerror("Training Error", "Face recognizer not available. Please check OpenCV installation.")
            return False

        try:
            # Train the model
            self.recognizer.train(faces, np.array(labels))

            # Save the trained model
            self.recognizer.save("trainer.yml")

            messagebox.showinfo("Success", f"Model trained successfully with {len(faces)} images from {len(set(labels))} students!")
            return True

        except Exception as e:
            messagebox.showerror("Training Error", f"Training failed: {str(e)}")
            return False

    def recognize_face(self, face_roi):
        """Recognize a face and return ID and confidence"""
        if self.recognizer is None:
            return -1, 0
        return self.recognizer.predict(face_roi)

    def real_time_face_detection(self):
        """Real-time face detection using webcam"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            messagebox.showinfo("Face Detection", "Face detection started. Press 'q' to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                faces, gray = self.detect_faces(frame)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, 'Face Detected', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                cv2.imshow('Face Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Error", f"Face detection error: {str(e)}")

    def real_time_face_recognition_attendance(self, attendance_system):
        """Real-time face recognition with automatic attendance marking"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            if self.recognizer is None:
                messagebox.showerror("Error", "Face recognizer not available. Please train the model first.")
                return

            messagebox.showinfo("Auto Attendance", "Automatic attendance started. Press 'q' to quit.")

            recognized_students = set()  # Track students already marked for today

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                faces, gray = self.detect_faces(frame)

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]

                    # Recognize the face
                    id, confidence = self.recognize_face(face_roi)

                    if confidence < 70:  # Confidence threshold
                        student = attendance_system.get_student_by_id(id)
                        if student:
                            student_key = f"{student['id']}_{datetime.date.today()}"
                            if student_key not in recognized_students:
                                # Mark attendance automatically
                                result = attendance_system.mark_automatic_attendance(id, confidence)
                                if result:
                                    recognized_students.add(student_key)
                                    print(f"Auto Attendance: {result}")

                            # Display recognized student
                            cv2.putText(frame, f"{student['name']}", (x, y - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {id}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, 'Unknown', (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, 'Unknown', (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.imshow('Automatic Attendance System', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Error", f"Face recognition error: {str(e)}")

    def detect_faces_in_image(self, image_path):
        """Detect faces in a static image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", "Could not load image")
                return

            faces, _ = self.detect_faces(img)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, 'Face Detected', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow('Face Detection in Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            messagebox.showinfo("Result", f"Found {len(faces)} face(s) in the image")

        except Exception as e:
            messagebox.showerror("Error", f"Image face detection error: {str(e)}")
