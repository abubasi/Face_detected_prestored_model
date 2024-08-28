import cv2
import os


haar_file = 'haarcascade_frontalface_default.xml'
dataset = 'dataset'
sub_data = str(input("Enter the name : "))
print(sub_data)
path = os.path.join(dataset, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)
    
(width, height) = (130, 100)
face_casecade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count < 50:   
    print(count)
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face = face_casecade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_region = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face_region, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(1)
        if key==27:
            break
webcam.release()
cv2.destroyAllWindows()
