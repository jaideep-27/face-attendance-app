import cv2
import numpy as np
from PIL import Image
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

def capture_images(name, student_id, save_dir="images"):
    """Capture face images for registration"""
    cam = None
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Try different camera indices with detailed error reporting
        camera_found = False
        for idx in range(3):  # Try indices 0, 1, and 2
            try:
                cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                if cam is not None and cam.isOpened():
                    # Try to read a test frame
                    ret, test_frame = cam.read()
                    if ret and test_frame is not None:
                        print(f"Successfully connected to camera {idx}")
                        camera_found = True
                        break
                    else:
                        print(f"Camera {idx} opened but couldn't read frame")
                        cam.release()
                else:
                    print(f"Failed to open camera {idx}")
            except Exception as e:
                print(f"Error trying camera {idx}: {str(e)}")
                if cam is not None:
                    cam.release()
                    cam = None
        
        if not camera_found:
            raise RuntimeError("No working camera found. Please check your camera connection and permissions.")
            
        count = 0
        max_attempts = 50  # Maximum frames to try
        attempts = 0
        
        print("Starting image capture. Please look at the camera...")
        
        while attempts < max_attempts:
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
                count += 1
                face_path = os.path.join(save_dir, f"{name}_{student_id}_{count}.jpg")
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(face_path, face_img)
                print(f"Captured image {count}/5")
                
            if count >= 5:  # Capture 5 face images
                print("Successfully captured all required images!")
                break
                
            attempts += 1
            
        if count == 0:
            print("Failed to capture any images. Please try again in better lighting conditions.")
            return False
            
        return count > 0
        
    except Exception as e:
        print(f"Error during image capture: {str(e)}")
        return False
    finally:
        if cam is not None:
            cam.release()

def train_model(image_dir, model_dir):
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_samples = []
        ids = []
        
        for image_path in os.listdir(image_dir):
            if not image_path.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img = Image.open(os.path.join(image_dir, image_path)).convert('L')
            img_np = np.array(img, 'uint8')
            student_id = int(image_path.split('_')[1])
            faces = detector.detectMultiScale(img_np)
            
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(student_id)
                
        if not face_samples:
            return False
            
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(os.path.join(model_dir, "trained_model.yml"))
        
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

def recognize_face(model_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = os.path.join(model_dir, "trained_model.yml")
    
    if not os.path.exists(model_path):
        return None
        
    recognizer.read(model_path)
    cam = cv2.VideoCapture(0)
    
    # Wait for camera to initialize
    for _ in range(5):
        ret, frame = cam.read()
        if not ret:
            return None
            
    # Capture and process frame
    ret, frame = cam.read()
    if not ret:
        return None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        student_id, confidence = recognizer.predict(face)
        
        if confidence < 70:
            cam.release()
            cv2.destroyAllWindows()
            return student_id
            
    cam.release()
    cv2.destroyAllWindows()
    return None

def get_user_by_face(model_dir="model"):
    """Recognize a face and return the student ID if found"""
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        model_path = os.path.join(model_dir, "trained_model.yml")
        
        if not os.path.exists(model_path):
            print("No trained model found")
            return None
            
        recognizer.read(model_path)
        cam = cv2.VideoCapture(0)
        
        if not cam.isOpened():
            print("Error: Could not open camera")
            return None
        
        # Wait for camera to initialize
        for _ in range(5):
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame during initialization")
                return None
                
        # Capture and process frame
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame")
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        student_id = None
        min_confidence = float('inf')
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(face)
            
            if confidence < 70 and confidence < min_confidence:
                student_id = id_
                min_confidence = confidence
        
        return student_id
        
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")
        return None
    finally:
        try:
            cam.release()
            cv2.destroyAllWindows()
        except:
            pass
