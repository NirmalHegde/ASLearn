import firebase_admin
from firebase_admin import credentials, firestore
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import time

cred = credentials.Certificate('./aslearn.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()

doc_ref = db.collection(u'sampleData').document(u'frontend')


def on_snapshot(doc_snapshot, changes, read_time):
    print(changes)


docWatch = doc_ref.on_snapshot(on_snapshot)

model = tf.keras.models.load_model('anjalis-model.h5')

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

arr = [65, 66, 72, 79, 85, 86, 89]
counter = 0
timer = 0

while True:
    _, image = cap.read(0)
    imgWebcam = cv2.flip(image, 1)

    img = cv2.cvtColor(imgWebcam, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, 0)
    predictions = model.predict(img)
    predictions = np.argmax(predictions)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imgWebcam, chr(predictions + 65), (10, 100), font, 3, (0, 255, 0), 10)
    cv2.putText(imgWebcam, "Show Me:", (330, 340), font, 2, (255, 0, 0), 10)
    cv2.putText(imgWebcam, chr(arr[counter]), (450, 420), font, 3, (255, 0, 0), 10)

    if (predictions + 65) == arr[counter]:
        cv2.putText(imgWebcam, "That is right!", (200, 200), font, 1, (0, 255, 0), 5)
        timer += 1
        if timer % 10 == 0:
            counter += 1
        if counter > 6:
            counter = 0

    cv2.imshow("Window", imgWebcam)
    cv2.waitKey(1)
