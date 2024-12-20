import cv2
import os
import numpy as np
from PIL import Image
import logging

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

try:
    # Initialize face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    try:
        # Try alternative import
        recognizer = cv2.createLBPHFaceRecognizer()
    except AttributeError:
        logger.error("Failed to create face recognizer. OpenCV face recognition module not available.")
        raise ImportError("OpenCV face recognition module not available. Please install opencv-contrib-python package.")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        try:
            # Convert image to grayscale
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Get the user ID from the image path
            user_id = int(os.path.split(image_path)[-1].split("_")[1].split(".")[0])
            
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(user_id)

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            continue

    return face_samples, ids

def train_model():
    try:
        logger.info("Training face recognition model...")
        faces, ids = get_images_and_labels('images')
        recognizer.train(faces, np.array(ids))
        recognizer.save('model/trained_model.yml')
        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

def capture_images(name, user_id):
    try:
        # Create directories if they don't exist
        if not os.path.exists('images'):
            os.makedirs('images')
        if not os.path.exists('model'):
            os.makedirs('model')

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open camera")
            return False

        count = 0
        while count < 30:  # Capture 30 images
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                # Save the captured face
                cv2.imwrite(f"images/User_{user_id}_{count}.jpg", gray[y:y+h, x:x+w])

        cap.release()
        return count > 0
    except Exception as e:
        logger.error(f"Error capturing images: {str(e)}")
        return False

def get_user_by_face():
    try:
        # Load the trained model
        if not os.path.exists('model/trained_model.yml'):
            logger.error("Error: Model file not found")
            return None

        recognizer.read('model/trained_model.yml')

        # Initialize camera
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
            face = gray[y:y+h, x:x+w]
            user_id, confidence = recognizer.predict(face)
            
            # Lower confidence means better match
            if confidence < 100:
                cap.release()
                return str(user_id)

        cap.release()
        return None
    except Exception as e:
        logger.error(f"Error recognizing face: {str(e)}")
        return None
