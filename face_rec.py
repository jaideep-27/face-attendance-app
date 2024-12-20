import cv2
import numpy as np
from PIL import Image
import os
import json
import logging
from download_cascade import download_cascade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download and initialize face detector
download_cascade()
cascade_path = os.path.join('model', 'haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(cascade_path)

def capture_images(name, user_id, save_dir="images"):
    """Capture multiple images of a face for registration"""
    try:
        # Create directory for user images
        user_dir = os.path.join(save_dir, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        # Initialize camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow backend on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Fallback to default
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            return False
            
        images_captured = 0
        required_images = 5
        
        while images_captured < required_images:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract and save face
                face = gray[y:y+h, x:x+w]
                face_path = os.path.join(user_dir, f"{images_captured}.jpg")
                cv2.imwrite(face_path, face)
                
                images_captured += 1
                logger.info(f"Captured image {images_captured}/{required_images}")
                break
            
            # Display frame
            cv2.imshow('Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        if images_captured == required_images:
            # Extract and save features
            features = extract_features(user_dir)
            save_features(user_id, features)
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error capturing images: {str(e)}")
        return False

def extract_features(image_dir):
    """Extract features from captured images"""
    features = []
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Calculate histogram as feature
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist.tolist())
    
    return np.mean(features, axis=0).tolist()

def save_features(user_id, features):
    """Save user features to JSON file"""
    features_file = os.path.join('model', 'features.json')
    
    try:
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                all_features = json.load(f)
        else:
            all_features = {}
        
        all_features[str(user_id)] = features
        
        with open(features_file, 'w') as f:
            json.dump(all_features, f)
            
    except Exception as e:
        logger.error(f"Error saving features: {str(e)}")

def get_user_by_face():
    """Recognize user from webcam"""
    try:
        features_file = os.path.join('model', 'features.json')
        if not os.path.exists(features_file):
            logger.error("No registered users found")
            return None
            
        with open(features_file, 'r') as f:
            all_features = json.load(f)
            
        if not all_features:
            logger.error("No registered users found")
            return None
            
        # Initialize camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow backend on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Fallback to default
            
        if not cap.isOpened():
            logger.error("Could not open camera")
            return None
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                
                # Calculate histogram
                hist = cv2.calcHist([face], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                # Compare with stored features
                min_dist = float('inf')
                matched_id = None
                
                for user_id, features in all_features.items():
                    dist = np.linalg.norm(hist - np.array(features))
                    if dist < min_dist and dist < 0.3:  # Threshold for matching
                        min_dist = dist
                        matched_id = user_id
                
                if matched_id:
                    cap.release()
                    cv2.destroyAllWindows()
                    return matched_id
                    
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return None
        
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}")
        return None
