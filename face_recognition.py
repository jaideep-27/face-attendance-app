import cv2
import numpy as np
import os
import urllib.request

# Download and initialize face detector
CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
CASCADE_PATH = os.path.join('model', 'haarcascade_frontalface_default.xml')

# Download cascade classifier if not exists
if not os.path.exists(CASCADE_PATH):
    print(f"Downloading cascade classifier to {CASCADE_PATH}")
    os.makedirs('model', exist_ok=True)
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)

detector = cv2.CascadeClassifier(CASCADE_PATH)
if detector.empty():
    raise RuntimeError(f"Failed to load cascade classifier from {CASCADE_PATH}")

# Initialize LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
RECOGNIZER_PATH = os.path.join('model', 'lbph_recognizer.yml')

def capture_images(name, student_id, save_dir="images"):
    """Capture face images for registration"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        student_dir = os.path.join(save_dir, f"{student_id}_{name}")
        os.makedirs(student_dir, exist_ok=True)
        
        # Try different camera indices
        cam = None
        for idx in range(2):
            try:
                cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cam is not None and cam.isOpened():
                    ret, test_frame = cam.read()
                    if ret and test_frame is not None:
                        print(f"Successfully connected to camera {idx}")
                        break
                    else:
                        print(f"Camera {idx} opened but couldn't read frame")
                        cam.release()
                        cam = None
            except Exception as e:
                print(f"Error trying camera {idx}: {str(e)}")
                if cam is not None:
                    cam.release()
                    cam = None
        
        if cam is None:
            raise RuntimeError("No working camera found. Please check your camera connection and permissions.")
            
        count = 0
        max_attempts = 50
        attempts = 0
        
        print("Starting image capture. Please look at the camera...")
        
        while attempts < max_attempts and count < 5:
            ret, frame = cam.read()
            if not ret:
                print(f"Failed to read frame (attempt {attempts + 1}/{max_attempts})")
                attempts += 1
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            if len(faces) == 0:
                print("No face detected. Please make sure your face is visible to the camera.")
                attempts += 1
                continue
            
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                count += 1
                cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face_img)
                print(f"Captured image {count}/5")
                break
                
            attempts += 1
        
        if count == 0:
            print("Failed to capture any images. Please try again in better lighting conditions.")
            return False
            
        print("Successfully captured all required images!")
        return True
        
    except Exception as e:
        print(f"Error during image capture: {str(e)}")
        return False
    finally:
        if cam is not None:
            cam.release()

def train_model(image_dir="images", model_dir="model"):
    """Train the face recognition model"""
    try:
        if not os.path.exists(image_dir):
            raise RuntimeError(f"Image directory {image_dir} does not exist")
            
        faces = []
        labels = []
        label_map = {}
        current_label = 0
        
        # Load training images
        for person_dir in os.listdir(image_dir):
            if os.path.isdir(os.path.join(image_dir, person_dir)):
                student_id = person_dir.split('_')[0]
                label_map[current_label] = student_id
                
                person_path = os.path.join(image_dir, person_dir)
                for img_name in os.listdir(person_path):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(person_path, img_name)
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(face_img)
                        labels.append(current_label)
                
                current_label += 1
        
        if not faces:
            raise RuntimeError("No training images found")
            
        # Train the recognizer
        recognizer.train(faces, np.array(labels))
        
        # Save the model and label mapping
        os.makedirs(model_dir, exist_ok=True)
        recognizer.save(RECOGNIZER_PATH)
        
        # Save label mapping
        label_map_path = os.path.join(model_dir, 'label_map.txt')
        with open(label_map_path, 'w') as f:
            for label, student_id in label_map.items():
                f.write(f"{label},{student_id}\n")
                
        print("Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

def get_user_by_face(frame=None):
    """Recognize a face and return the student ID"""
    try:
        if not os.path.exists(RECOGNIZER_PATH):
            raise RuntimeError("Face recognition model not found. Please train the model first.")
            
        # Load label mapping
        label_map_path = os.path.join('model', 'label_map.txt')
        label_map = {}
        with open(label_map_path, 'r') as f:
            for line in f:
                label, student_id = line.strip().split(',')
                label_map[int(label)] = student_id
        
        # Load the recognizer
        recognizer.read(RECOGNIZER_PATH)
        
        if frame is None:
            # Try to capture a frame from the camera
            cam = None
            for idx in range(2):
                try:
                    cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cam is not None and cam.isOpened():
                        ret, frame = cam.read()
                        if ret and frame is not None:
                            break
                except:
                    if cam is not None:
                        cam.release()
                        cam = None
            
            if frame is None:
                raise RuntimeError("Could not capture frame from camera")
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return None
            
        # For each detected face
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            
            # Predict the label
            label, confidence = recognizer.predict(face_img)
            
            # If confidence is too low, return None
            if confidence > 100:  # Adjust this threshold as needed
                continue
                
            # Return the student ID
            return label_map.get(label)
        
        return None
        
    except Exception as e:
        print(f"Error during face recognition: {str(e)}")
        return None
    finally:
        if 'cam' in locals() and cam is not None:
            cam.release()
