import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
# import time
from gtts import gTTS
# from pydub import AudioSegment
import streamlit as st
import io

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)  # Reset the file pointer to the beginning
    return audio_bytes

# Function to classify hand gestures
def classify_gesture(img, detector, classifier):
    offset = 20 
    imgSize = 400
    folder = "Data/hello"
    labels = ['hello', 'thankyou', 'yes', 'no', 'iloveyou', 'please']
    x=0
    y=0
    w=0
    h=0
    
    imgOutput = img.copy()
    hands, _ = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
    else:
        index=0

    cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 2)
    
    return imgOutput, labels[index], x, y

st.title("Sign Language Interpreter")

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables
gesture = ""
start_x = -1

# Main loop
while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.write("Error: Couldn't read frame from webcam")
        break

    if start_x == -1:
        imgOutput, gesture, start_x, _ = classify_gesture(img, detector, classifier)

    elif start_x > -1:
        imgOutput, gesture, _, _ = classify_gesture(img, detector, classifier)

        # Play speech when 's' is pressed
        if st.button("Play Speech"):
            st.audio(text_to_speech(gesture), format="audio/wav")

        
        # Break the loop when 'Quit' button is pressed
        if st.button("Quit"):
            break

        if gesture == 'hello':
            break
