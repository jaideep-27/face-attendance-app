import cv2
import os
import numpy as np
from PIL import Image
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the face cascade classifier
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_path):
    import urllib.request
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise ValueError("Error loading cascade classifier")

class SimpleFaceRecognizer:
    def __init__(self):
        self.faces_dir = 'faces'
        self.model_file = 'model/face_data.json'
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs('model', exist_ok=True)
        self.face_data = self._load_face_data()

    def _load_face_data(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_face_data(self):
        with open(self.model_file, 'w') as f:
            json.dump(self.face_data, f)

    def _get_face_features(self, face_img):
        # Resize to standard size for comparison
        face_img = cv2.resize(face_img, (100, 100))
        # Calculate histogram as a simple feature
        hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.tolist()

    def _compare_features(self, features1, features2):
        return cv2.compareHist(
            np.array(features1, dtype=np.float32),
            np.array(features2, dtype=np.float32),
            cv2.HISTCMP_CORREL
        )

# Initialize the recognizer
recognizer = SimpleFaceRecognizer()

def train_model():
    """Train the face recognition model"""
    try:
        logger.info("Training face recognition model...")
        # The model is automatically updated when new faces are added
        return True
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

def capture_images(name, user_id):
    """Capture face images for registration"""
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open camera")
            return False

        count = 0
        face_features = []
        
        while count < 5:  # Capture 5 images
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                features = recognizer._get_face_features(face_img)
                face_features.append(features)
                
                # Save the face image
                img_path = os.path.join(recognizer.faces_dir, f"{user_id}_{count}.jpg")
                cv2.imwrite(img_path, face_img)
                
                count += 1
                break  # Only use the first face detected

        cap.release()

        if count > 0:
            # Save face features
            recognizer.face_data[str(user_id)] = {
                'name': name,
                'features': face_features,
                'timestamp': datetime.now().isoformat()
            }
            recognizer._save_face_data()
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error capturing images: {str(e)}")
        return False

def get_user_by_face():
    """Recognize a face and return the user ID"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open camera")
            return None

        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Could not read frame")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            features = recognizer._get_face_features(face_img)
            
            best_match = None
            best_score = -1
            
            for user_id, data in recognizer.face_data.items():
                for stored_features in data['features']:
                    score = recognizer._compare_features(features, stored_features)
                    if score > best_score:
                        best_score = score
                        best_match = user_id

            cap.release()
            
            # Threshold for face recognition confidence
            if best_score > 0.5:
                return best_match
                
        cap.release()
        return None
    except Exception as e:
        logger.error(f"Error recognizing face: {str(e)}")
        return None
