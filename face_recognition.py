import cv2
import numpy as np
from PIL import Image
import os

# Initialize face detector
cascade_path = os.path.join('model', 'haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    raise RuntimeError("Error: Could not load face detection cascade classifier")

def capture_images(name, student_id, save_dir="images"):
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open camera")
            return False
            
        count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                count += 1
                face_path = os.path.join(save_dir, f"{name}_{student_id}_{count}.jpg")
                cv2.imwrite(face_path, gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            cv2.imshow("Face Capture - Press 'q' to finish", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
                break
                
        cam.release()
        cv2.destroyAllWindows()
        
        return count > 0
        
    except Exception as e:
        print(f"Error capturing images: {str(e)}")
        return False
    finally:
        try:
            cam.release()
            cv2.destroyAllWindows()
        except:
            pass

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
