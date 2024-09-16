import cv2
import numpy as np
from statistics import mode
from keras.api.models import load_model 
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# Parameters for loading data and images
detection_model_path = 'models/haarcascade_frontalface_alt2.xml'
emotion_model_path = 'models/fer2013_mini_XCEPTION.107-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)

# Loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Function to process a single image
def process_image(image_path):
    bgr_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    
    emotion_window = []

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        
        # Increase text size by changing font scale and thickness
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 2, 2)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # Create a window
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
    # Resize the window to the desired dimensions (width, height)
    cv2.resizeWindow('Emotion Detection', 800, 600)
    cv2.imshow('Emotion Detection', bgr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage: replace 'image_path.jpg' with your image file path
process_image('images/sad-person.jpeg')
