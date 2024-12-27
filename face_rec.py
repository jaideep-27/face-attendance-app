import cv2
import numpy as np
from PIL import Image
import os
import json
import logging
import streamlit as st
from download_cascade import download_cascade

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download and initialize face detector
download_cascade()
cascade_path = os.path.join('model', 'haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(cascade_path)

def train_model(user_id, user_dir):
    """Train the face recognition model for a new user"""
    try:
        logger.info(f"Training model for user {user_id}")
        
        # Load and preprocess face image
        face_path = os.path.join(user_dir, "0.jpg")
        if not os.path.exists(face_path):
            logger.error(f"Face image not found at {face_path}")
            return False
            
        # Extract features
        features = extract_features(user_dir)
        if features is None:
            logger.error("Failed to extract features")
            return False
            
        # Save features
        if not save_features(user_id, features):
            logger.error("Failed to save features")
            return False
            
        logger.info("Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        return False

def capture_images(name, user_id, save_dir="images"):
    """Capture multiple images of a face for registration"""
    try:
        logger.info(f"Starting image capture for user: {name} (ID: {user_id})")
        
        # Clean up any existing files for this user
        user_dir = os.path.join(save_dir, str(user_id))
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
        
        # Create directory for user images
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"Created user directory: {user_dir}")
        
        # For cloud deployment, use Streamlit's camera input
        img_file = st.camera_input(
            label=f"Take a picture for {name}",
            key=f"camera_{user_id}",
            help="Please look directly at the camera and ensure good lighting"
        )
        
        while img_file is None:
            logger.info("Waiting for user to take photo...")
            img_file = st.camera_input(
                label=f"Take a picture for {name}",
                key=f"camera_{user_id}",
                help="Please look directly at the camera and ensure good lighting"
            )
        
        logger.info("Image captured from camera")
        
        # Save the captured image
        img = Image.open(img_file)
        img = np.array(img)
        logger.info(f"Image shape: {img.shape}")
        
        # Convert to grayscale if image is RGB
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            logger.info("Converted RGB image to grayscale")
        else:
            gray = img
            logger.info("Image is already grayscale")
            
        # Detect faces
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        logger.info(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            logger.error("No face detected in the image")
            st.error("No face detected in the image. Please try again.")
            return False
            
        if len(faces) > 1:
            logger.error("Multiple faces detected")
            st.error("Multiple faces detected. Please ensure only one person is in the frame.")
            return False
            
        # Extract and save face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face_path = os.path.join(user_dir, "0.jpg")
        cv2.imwrite(face_path, face)
        logger.info(f"Saved face image to: {face_path}")
        
        # Calculate and save features immediately
        hist = cv2.calcHist([face], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten().tolist()
        
        # Save features
        os.makedirs('model', exist_ok=True)
        features_file = os.path.join('model', 'features.json')
        
        try:
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    features = json.load(f)
            else:
                features = {}
        except:
            features = {}
            
        features[str(user_id)] = hist
        
        with open(features_file, 'w') as f:
            json.dump(features, f)
            
        logger.info("Face features saved successfully")
        st.success("✨ Face registered successfully!")
        return True
            
    except Exception as e:
        logger.error(f"Error capturing images: {str(e)}")
        st.error(f"An error occurred during image capture: {str(e)}")
        return False

def extract_features(image_dir):
    """Extract features from captured images"""
    try:
        logger.info(f"Extracting features from images in: {image_dir}")
        features = []
        for img_name in os.listdir(image_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(image_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.error(f"Failed to load image: {img_path}")
                    return None
                    
                logger.info(f"Processing image: {img_path}")
                
                # Calculate histogram as feature
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features.append(hist.tolist())
                logger.info("Calculated histogram features")
        
        if not features:
            logger.error("No features extracted")
            return None
            
        mean_features = np.mean(features, axis=0).tolist()
        logger.info("Calculated mean features")
        logger.info(f"Extracted features: {len(features)} images processed.")
        return mean_features
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}", exc_info=True)
        return None

def save_features(user_id, features):
    """Save user features to JSON file"""
    try:
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        features_file = os.path.join('model', 'features.json')
        logger.info(f"Saving features for user {user_id} to {features_file}")
        
        # Load existing features or create new dictionary
        if os.path.exists(features_file):
            try:
                with open(features_file, 'r') as f:
                    all_features = json.load(f)
                    logger.info("Loaded existing features file")
            except json.JSONDecodeError:
                logger.warning("Corrupted features file, creating new one")
                all_features = {}
        else:
            all_features = {}
            logger.info("Created new features dictionary")
        
        # Save the features
        all_features[str(user_id)] = features
        with open(features_file, 'w') as f:
            json.dump(all_features, f)
            logger.info("Successfully saved features to file")
            
        return True
            
    except Exception as e:
        logger.error(f"Error saving features: {str(e)}", exc_info=True)
        st.error("An error occurred while saving user data. Please try again.")
        return False

def get_user_by_face():
    """Recognize user from webcam"""
    try:
        logger.info("Starting face recognition")
        
        # Check if features file exists and has data
        features_file = os.path.join('model', 'features.json')
        if not os.path.exists(features_file):
            logger.error("No registered users found (features file missing)")
            st.error("No registered users found. Please register first.")
            return None
            
        try:
            with open(features_file, 'r') as f:
                all_features = json.load(f)
        except json.JSONDecodeError:
            logger.error("Corrupted features file")
            st.error("User data is corrupted. Please register again.")
            return None
            
        if not all_features:
            logger.error("No registered users found (empty features)")
            st.error("No registered users found. Please register first.")
            return None
            
        # For cloud deployment, use Streamlit's camera input
        img_file = st.camera_input(
            label="Take a picture for recognition",
            key="recognition_camera",
            help="Please look directly at the camera and ensure good lighting"
        )
        
        if img_file is None:
            logger.info("Waiting for user to take photo...")
            return None
            
        logger.info("Image captured from camera for recognition")
        
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
            logger.error("No face detected in the image")
            st.error("No face detected in the image. Please try again.")
            return None
            
        if len(faces) > 1:
            logger.error("Multiple faces detected")
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
            logger.info(f"Distance from user {user_id}: {dist}")
            if dist < min_dist and dist < 0.5:  # Increased threshold
                min_dist = dist
                matched_id = user_id
        
        if matched_id:
            logger.info(f"Face recognized as user {matched_id} with distance {min_dist}")
            st.success("✨ Face recognized successfully!")
            return matched_id
        else:
            logger.error(f"Face not recognized (min distance: {min_dist})")
            st.error("Face not recognized. Please try again or register if you're a new user.")
            return None
            
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}")
        st.error("An error occurred during face recognition. Please try again.")
        return None
