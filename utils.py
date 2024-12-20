import cv2
import os
import sqlite3
import time

def clear_inputs():
    pass  # Streamlit handles clearing inputs automatically

def get_user_by_face(model_path):
    """
    Returns the user_id if face is recognized, None otherwise
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = f"{model_path}/trained_model.yml"
    
    if not os.path.exists(model_path):
        return None
    
    recognizer.read(model_path)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    cam = cv2.VideoCapture(0)
    
    # Wait for camera to initialize
    time.sleep(1)
    
    user_id = None
    max_attempts = 30  # Reduced attempts for faster response
    attempts = 0
    
    while attempts < max_attempts:
        ret, frame = cam.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)  # More lenient parameters
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            
            # Ensure face is large enough
            if w < 100 or h < 100:
                continue
                
            predicted_id, confidence = recognizer.predict(face)
            
            if confidence < 80:  # More lenient confidence threshold
                user_id = predicted_id
                break
                
        if user_id is not None:
            break
            
        attempts += 1
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    return user_id
