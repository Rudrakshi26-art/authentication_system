import cv2
import numpy as np
import math
from tkinter import messagebox

class GestureRecognition:
    def __init__(self):
        # Skin color detection parameters
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def detect_hand_gesture(self, frame):
        """Detect basic hand gestures"""
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create skin mask
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Contours found: {len(contours)}")  # Debugging

        gesture = "No Gesture"

        for contour in contours:
            if cv2.contourArea(contour) > 2000:  # Minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate aspect ratio
                aspect_ratio = float(w) / h

                # Calculate convexity defects for finger counting
                hull = cv2.convexHull(contour, returnPoints=False)
                if len(hull) > 3:
                    defects = cv2.convexityDefects(contour, hull)
                    if defects is not None:
                        finger_count = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])

                            # Calculate angle
                            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                            if angle <= 90 and d > 10000:
                                finger_count += 1

                        if finger_count == 0:
                            gesture = "Fist"
                        elif finger_count == 1:
                            gesture = "One Finger"
                        elif finger_count == 2:
                            gesture = "Two Fingers"
                        elif finger_count >= 3:
                            gesture = "Open Hand"

                # Draw bounding rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if gesture != "No Gesture":
            print(f"Detected gesture: {gesture}")  # Debugging

        return frame, gesture

    def real_time_gesture_recognition(self):
        """Real-time gesture recognition"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            messagebox.showinfo("Gesture Recognition", "Gesture recognition started. Press 'q' to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Detect gesture
                processed_frame, gesture = self.detect_hand_gesture(frame)

                # Display gesture
                cv2.putText(processed_frame, f"Gesture: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Gesture Recognition', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Error", f"Gesture recognition error: {str(e)}")

    def gesture_controlled_attendance(self, attendance_system, face_recognizer):
        """Gesture-controlled attendance system"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            messagebox.showinfo("Gesture Control", "Gesture-controlled attendance started. Use gestures to control. Press 'q' to quit.")

            current_student = None
            gesture_buffer = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame horizontally
                frame = cv2.flip(frame, 1)

                # Detect gesture
                processed_frame, gesture = self.detect_hand_gesture(frame)

                # Add gesture to buffer for stability
                gesture_buffer.append(gesture)
                if len(gesture_buffer) > 10:
                    gesture_buffer.pop(0)

                # Get most common gesture in buffer
                if gesture_buffer:
                    stable_gesture = max(set(gesture_buffer), key=gesture_buffer.count)
                else:
                    stable_gesture = gesture

                # Face recognition
                faces, gray = face_recognizer.detect_faces(processed_frame)

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    id, confidence = face_recognizer.recognize_face(face_roi)

                    if confidence < 70:
                        student = attendance_system.get_student_by_id(id)
                        if student:
                            current_student = student
                            cv2.putText(processed_frame, f"Student: {student['name']}", (x, y - 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        else:
                            cv2.putText(processed_frame, 'Unknown', (x, y - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Gesture control
                if stable_gesture == "Open Hand" and current_student:
                    # Mark attendance
                    result = attendance_system.mark_automatic_attendance(current_student['id'], 0.0)
                    if result:
                        cv2.putText(processed_frame, "ATTENDANCE MARKED!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        print(f"Gesture Attendance: {result}")
                        current_student = None  # Reset after marking

                elif stable_gesture == "Fist":
                    cv2.putText(processed_frame, "CANCEL", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    current_student = None

                # Display current gesture
                cv2.putText(processed_frame, f"Gesture: {stable_gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if current_student:
                    cv2.putText(processed_frame, f"Ready to mark: {current_student['name']}", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow('Gesture Controlled Attendance', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Error", f"Gesture control error: {str(e)}")
