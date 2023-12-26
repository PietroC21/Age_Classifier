import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf    
from keras.models import load_model            
from keras.models import Sequential
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

#Load the karas model
model = tf.keras.models.load_model('CNN.h5')
img = cv2.imread('trans.jpg')
img = cv2.resize(img, (256, 256))
img = np.expand_dims(img/255,0)
yhat = model.predict(img)
print(yhat)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('CNN.h5')
cap = cv2.VideoCapture(0)
# Set the width and height of the frames to 256x256
width, height = 256, 256
cap.set(3, width)  # 3 corresponds to CV_CAP_PROP_FRAME_WIDTH
cap.set(4, height)  # 4 corresponds to CV_CAP_PROP_FRAME_HEIGHT


# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
i = 0
# Continuously capture frames from the camera
while True:
    i+=1
    # Read a frame from the camera
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Could not read frame.")
        break
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        resized_frame = cv2.resize(frame, (256, 256))
        yhat = model.predict(np.expand_dims(resized_frame/255,0))
        print(yhat)

        sex = np.where(yhat>0.5, 1, 0)
        if sex == 1:
            cv2.putText(frame, f'Sex: Woman {100*yhat[0][0]:.2f}%' , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else: 
            cv2.putText(frame, f'Sex: Man {100*(1-yhat[0][0]):.2f}%' , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)



    # Display the result
    cv2.imshow('Detected Faces', frame)
    #print(result)


    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
