import cv2
import os
import csv
import yt_dlp
from ultralytics import YOLO
from collections import defaultdict

emotions_count = defaultdict(int)

def get_video_url(youtube_url):
    ydl_opts = {
        'format': 'best',  
        'quiet': True,    
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get("url", None)
    return video_url

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            emotions_count[result.names[int(box.cls[0])]] += 1
    return img, results

raw_url = input("\nInsira a URL do vídeo: ")
video_url = get_video_url(raw_url)
model = YOLO("notebook/runs/detect/train/weights/best.pt")
video_capture = cv2.VideoCapture(video_url)

while True:
    success, frame = video_capture.read()

    if not success or frame is None:
        print("Fim do video")
        break
    
    result_img, _ = predict_and_detect(model, frame)
    
    cv2.imshow("Image", result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()

file_name = "yolov8_emotions.csv"
file_save_path = f'./src/emotions_count/{file_name}'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

with open(file_save_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Emotion", "Count"])  # header row
    for emotion, count in emotions_count.items():
        writer.writerow([emotion, count])

print(f"Contagem de emoções salva no arquivo: {file_name}")
