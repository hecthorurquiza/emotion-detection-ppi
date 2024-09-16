import cv2
from deepface import DeepFace

# Load the face detection model
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')

# Read the image
image = cv2.imread('images/sad-person.jpeg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit(1)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to RGB format (if needed for other operations)
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Loop over detected faces
for (x, y, w, h) in faces:
    # Extract the face ROI (Region of Interest)
    face_roi = rgb_image[y:y + h, x:x + w]

    # Perform emotion analysis on the face ROI
    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    
    # Determine the dominant emotion
    emotion = result[0]['dominant_emotion']
    
    # Draw a rectangle around the face and label with the predicted emotion
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Create a window
cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)

# Resize the window to the desired dimensions (width, height)
cv2.resizeWindow('Emotion Detection', 800, 600)

# Display the resulting image
cv2.imshow('Emotion Detection', image)

# Hold the window open until 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
