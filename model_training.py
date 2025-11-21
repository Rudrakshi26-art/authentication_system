import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from utils import Utils

class ModelTrainingWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Model Training")
        self.window.geometry("600x500")
        self.window.configure(bg='#2c3e50')
        
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Training Information", padding=10)
        info_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=15, width=70)
        self.info_text.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(info_frame, orient='vertical', command=self.info_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill='x', pady=(0, 10))
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill='x')
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to train")
        self.progress_label.pack()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(button_frame, text="Train Face Model", 
                  command=self.train_face_model).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Train Gesture Model", 
                  command=self.train_gesture_model).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Train Both", 
                  command=self.train_both_models).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", 
                  command=self.window.destroy).pack(side='right', padx=5)
    
    def log_message(self, message):
        self.info_text.insert('end', message + '\n')
        self.info_text.see('end')
        self.window.update()
    
    def train_face_model(self):
        try:
            self.log_message("=== Starting Face Model Training ===")
            
            faces = []
            labels = []
            employee_dirs = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]
            
            if not employee_dirs:
                self.log_message("❌ No employee data found for training")
                return
            
            total_images = 0
            for emp_dir in employee_dirs:
                face_dir = os.path.join('data', emp_dir, 'faces')
                if os.path.exists(face_dir):
                    face_files = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]
                    total_images += len(face_files)
                    self.log_message(f"Found {len(face_files)} face images for employee {emp_dir}")
            
            if total_images == 0:
                self.log_message("❌ No face images found for training")
                return
            
            self.progress['maximum'] = total_images
            self.progress['value'] = 0
            
            for emp_dir in employee_dirs:
                face_dir = os.path.join('data', emp_dir, 'faces')
                if os.path.exists(face_dir):
                    for face_file in os.listdir(face_dir):
                        if face_file.endswith('.jpg'):
                            img_path = os.path.join(face_dir, face_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                
                                faces.append(gray)
                                labels.append(int(emp_dir))
                                
                                self.progress['value'] += 1
                                self.progress_label['text'] = f"Processing {self.progress['value']}/{total_images}"
                                self.window.update()
                            else:
                                self.log_message(f"⚠️ Could not read image: {img_path}")
            
            if len(faces) == 0:
                self.log_message("❌ No valid face images found for training")
                return
            
            # Train LBPH face recognizer
            self.log_message("Training LBPH face recognizer...")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(labels))
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model_path = 'models/face_trainer.yml'
            recognizer.save(model_path)
            
            self.log_message(f"✅ Face model trained successfully!")
            self.log_message(f"📊 Employees: {len(employee_dirs)}")
            self.log_message(f"📊 Total face images: {len(faces)}")
            self.log_message(f"💾 Model saved to: {model_path}")
            
        except Exception as e:
            logging.error(f"Face training error: {str(e)}")
            self.log_message(f"❌ Error: {str(e)}")
    
    def train_gesture_model(self):
        try:
            self.log_message("=== Starting Gesture Model Training ===")
            
            gestures = []
            labels = []
            employee_dirs = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]
            
            if not employee_dirs:
                self.log_message("❌ No employee data found for training")
                return
            
            self.progress['maximum'] = len(employee_dirs)
            self.progress['value'] = 0
            
            gesture_count = 0
            for emp_dir in employee_dirs:
                gesture_file = os.path.join('data', emp_dir, 'gestures', 'gesture.pkl')
                if os.path.exists(gesture_file):
                    try:
                        with open(gesture_file, 'rb') as f:
                            gesture_data = pickle.load(f)
                        
                        if 'landmarks' in gesture_data:
                            landmarks = gesture_data['landmarks']
                            
                            # Validate landmarks format
                            if isinstance(landmarks, list) and len(landmarks) > 0:
                                gestures.append(landmarks)
                                labels.append(int(emp_dir))
                                gesture_count += 1
                                self.log_message(f"✅ Loaded gesture for employee {emp_dir}")
                            else:
                                self.log_message(f"⚠️ Invalid landmarks format for {emp_dir}")
                        else:
                            self.log_message(f"⚠️ No landmarks found for {emp_dir}")
                    
                    except Exception as e:
                        self.log_message(f"❌ Error loading gesture for {emp_dir}: {str(e)}")
                else:
                    self.log_message(f"⚠️ No gesture file found for {emp_dir}")
                
                self.progress['value'] += 1
                self.progress_label['text'] = f"Processing {self.progress['value']}/{len(employee_dirs)}"
                self.window.update()
            
            if len(gestures) == 0:
                self.log_message("❌ No valid gesture data found for training")
                return
            
            if len(gestures) < 2:
                self.log_message("❌ Need at least 2 employees with gestures for training")
                return
            
            self.log_message(f"📊 Total gesture samples: {len(gestures)}")
            self.log_message(f"📊 Unique employees: {len(set(labels))}")
            
            # Convert to numpy arrays
            X = np.array(gestures)
            y = np.array(labels)
            
            self.log_message(f"📊 Feature matrix shape: {X.shape}")
            self.log_message(f"📊 Labels shape: {y.shape}")
            
            # If only one sample per class, we can't do train_test_split
            if len(set(labels)) == len(gestures):
                self.log_message("ℹ️ Only one sample per class - using all data for training")
                X_train, y_train = X, y
                X_test, y_test = X, y
            else:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            # Train KNN classifier
            n_neighbors = min(3, len(set(y_train)))
            self.log_message(f"Training KNN classifier with n_neighbors={n_neighbors}")
            
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            
            # Evaluate if we have test data
            if len(X_test) > 0:
                y_pred = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                self.log_message(f"📊 Model accuracy: {accuracy:.2f}")
            else:
                accuracy = 1.0  # Perfect accuracy when using all data
                self.log_message("ℹ️ Using all data for training - accuracy not calculated")
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model_path = 'models/gesture_data.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': knn,
                    'accuracy': accuracy,
                    'feature_names': [f'landmark_{i}' for i in range(len(gestures[0]))],
                    'employees_trained': list(set(labels)),
                    'training_date': np.datetime64('now')
                }, f)
            
            self.log_message(f"✅ Gesture model trained successfully!")
            self.log_message(f"💾 Model saved to: {model_path}")
            self.log_message(f"👥 Employees in model: {list(set(labels))}")
            
        except Exception as e:
            logging.error(f"Gesture training error: {str(e)}")
            self.log_message(f"❌ Error: {str(e)}")
            import traceback
            self.log_message(f"❌ Traceback: {traceback.format_exc()}")
    
    def train_both_models(self):
        self.log_message("=== Training Both Models ===")
        self.train_face_model()
        self.train_gesture_model()
