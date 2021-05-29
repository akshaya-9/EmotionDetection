import cv2  
from keras.models import model_from_json  
from keras.preprocessing import image  
  

model = model_from_json(open("model.json", "r").read())  

model.load_weights('model.h5')  


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('images.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey()