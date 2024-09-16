import cv2
import yt_dlp
import csv
import os
from deepface import DeepFace
from collections import defaultdict

def get_video_url(youtube_url):
    ydl_opts = {
        'format': 'best',  
        'quiet': True,    
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get("url", None)
    return video_url

raw_url = input("\nInsira a URL do vídeo: ")
video_url = get_video_url(raw_url)
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt2.xml')
video_capture = cv2.VideoCapture(video_url)
emotions_count = defaultdict(int)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        emotions_count[emotion] += 1
        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture and close all windows
video_capture.release()
cv2.destroyAllWindows()

file_name = "deepface_emotions.csv"
file_save_path = f'./src/emotions_count/{file_name}'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

with open(file_save_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Emotion", "Count"])  # header row
    for emotion, count in emotions_count.items():
        writer.writerow([emotion, count])

print(f"Contagem de emoções salva no arquivo: {file_name}")