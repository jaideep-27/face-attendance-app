import cv2
import numpy as np
from PIL import Image
import os
import json
import logging
import streamlit as st
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
        
        # For cloud deployment, use Streamlit's camera input
        img_file = st.camera_input("Take a picture")
        
        if img_file is not None:
            # Save the captured image
            img = Image.open(img_file)
            img = np.array(img)
            
            # Convert to grayscale if image is RGB
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
                
            # Detect faces
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                st.error("No face detected in the image. Please try again.")
                return False
                
            if len(faces) > 1:
                st.error("Multiple faces detected. Please ensure only one person is in the frame.")
                return False
                
            # Extract and save face
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            face_path = os.path.join(user_dir, "0.jpg")
            cv2.imwrite(face_path, face)
            
            # Extract and save features
            features = extract_features(user_dir)
            save_features(user_id, features)
            
            st.success("Registration successful!")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error capturing images: {str(e)}")
        st.error("An error occurred during image capture. Please try again.")
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
        st.error("An error occurred while saving user data. Please try again.")

def get_user_by_face():
    """Recognize user from webcam"""
    try:
        features_file = os.path.join('model', 'features.json')
        if not os.path.exists(features_file):
            logger.error("No registered users found")
            st.error("No registered users found. Please register first.")
            return None
            
        with open(features_file, 'r') as f:
            all_features = json.load(f)
            
        if not all_features:
            logger.error("No registered users found")
            st.error("No registered users found. Please register first.")
            return None
            
        # For cloud deployment, use Streamlit's camera input
        img_file = st.camera_input("Take a picture for recognition")
        
        if img_file is not None:
            # Process the captured image
            img = Image.open(img_file)
            img = np.array(img)
            
            # Convert to grayscale if image is RGB
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
                
            # Detect faces
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                st.error("No face detected in the image. Please try again.")
                return None
                
            if len(faces) > 1:
                st.error("Multiple faces detected. Please ensure only one person is in the frame.")
                return None
                
            # Extract face and calculate features
            x, y, w, h = faces[0]
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
                st.success("Face recognized successfully!")
                return matched_id
            else:
                st.error("Face not recognized. Please try again or register if you're a new user.")
                return None
                
        return None
        
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}")
        st.error("An error occurred during face recognition. Please try again.")
        return None
