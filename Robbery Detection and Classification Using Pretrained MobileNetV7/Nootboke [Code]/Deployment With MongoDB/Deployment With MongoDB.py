import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog

model = load_model(r"C:\Users\Tarek Mohamed\Shop Lifting Detector.h5")

# Load MobileNet SSD for person detection
prototxt_path = r"D:\Downloads\MobileNetSSD_deploy.prototxt"
caffemodel_path = r"D:\Downloads\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

fake_alert_image_path = r"C:\Users\Tarek Mohamed\Fake Alert Image.png"

def extractFrames(videoPath, frameCount=8):
    cap = cv2.VideoCapture(videoPath)
    frames = []
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in np.linspace(0, totalFrames - 1, frameCount).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if success:
            frames.append(frame)
    cap.release()
    return frames

# Function to detect persons
def detectPerson(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    best_confidence = 0
    best_box = None
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        
        if confidence > 0.25 and class_id == 15:  
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = (startX, startY, endX, endY)
    
    return best_box

# Function to classify frames and highlight detected theft
def classifyAndShow(videoPath):
    frames = extractFrames(videoPath)
    resized_frames = [cv2.resize(frame, (180, 180)) for frame in frames]
    predictions = model.predict(np.array(resized_frames) / 255.0)
    
    detected = (predictions[:, 0] > 0.25).astype(int)  # Theft threshold
    lift_frame_indices = np.where(detected == 1)[0]  
    
    for lift_frame_index in lift_frame_indices:
        lift_frame = frames[lift_frame_index]
        
        # Detect person
        person_box = detectPerson(lift_frame)
        if person_box:
            (startX, startY, endX, endY) = person_box
            
            
            cv2.rectangle(lift_frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
            
            pad = 20  
            cropped_frame = lift_frame[max(0, startY-pad):min(lift_frame.shape[0], endY+pad),
                                       max(0, startX-pad):min(lift_frame.shape[1], endX+pad)]
            

            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            
            # Show the detected frame
            plt.figure(figsize=(6, 6))
            plt.imshow(cropped_frame)
            plt.title("Shoplifting Detected")
            plt.axis("off")
            plt.show()
            return  
    
    
    fake_alert = cv2.imread(fake_alert_image_path)
    fake_alert = cv2.cvtColor(fake_alert, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(fake_alert)
    plt.title("No Shoplifting Detected")
    plt.axis("off")
    plt.show()


def selectVideo():
    root = tk.Tk()
    root.withdraw()
    videoPath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if videoPath:
        classifyAndShow(videoPath)

selectVideo()
