import cv2
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Haar cascade for face detection
haar_file = 'haarcascade_frontalface_default.xml'
datasets = "dataset"  # Directory containing the dataset of faces
confidence_threshold = 80  # Adjustable threshold for recognition
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

# Load images and labels from the dataset
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            try:
                image = cv2.imread(path, 0)  # Load image in grayscale
                if image is not None:
                    images.append(image)
                    labels.append(int(id))  # Use 'id' as the label
                else:
                    logging.warning(f"Image not found: {path}")
            except Exception as e:
                logging.error(f"Error loading image {path}: {e}")
        id += 1

(width, height) = (130, 100)
(images, labels) = [np.array(lis) for lis in (images, labels)]

# Ensure labels are of integer type (CV_32SC1)
labels = labels.astype(np.int32)

# Create and train the LBPH recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)  # Use 0 for the primary webcam

if not webcam.isOpened():
    logging.error("Could not open webcam.")
    exit()

logging.info("Starting video stream...")

while True:
    _, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)

        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Check if the prediction confidence is below the threshold
        if prediction[1] < confidence_threshold:
            name = names[prediction[0]]
            cv2.putText(im, name, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            logging.info(f"Recognized: {name}")
        else:
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            logging.info("Unknown Person")
    
    cv2.imshow('Face Recognition', im)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
logging.info("Video stream ended.")
