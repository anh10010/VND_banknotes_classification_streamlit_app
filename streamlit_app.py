from numpy.core.fromnumeric import shape
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import time

model = tf.keras.models.load_model('MODEL_SAVE\my_model_checkpoint_Den_new.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
money_type = ['1000', '10000', '100000', '2000', '20000', '200000', '5000', '50000', '500000']

st.header('PREDICT VIETNAMESE DONG')
menu = ['Take photo by webcam', 'Upload a photo']
choice = st.sidebar.selectbox('Ways to input image', menu)

if choice == 'Take photo by webcam':
    st.write('Click on "Capture" button')
    cam = cv2.VideoCapture(0) # device 0. If not work, try with 1 or 2  

    capture_button = st.button('Capture')
    flip_checkbox = st.checkbox('Flip Camera')
    captured_image = np.array(None)

    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if flip_checkbox:
            #time.sleep(5)
            frame = cv2.flip(frame, 1)

        FRAME_WINDOW.image(frame)
        
        if capture_button:
            captured_image = frame
            img = cv2.resize(captured_image, (224,224))
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            index = np.argmax(prediction[0])
            money = money_type[index]
            st.write('This is:', money, 'VND')
            #FRAME_WINDOW.image(frame)
            break
    cam.release()
    
    

elif choice == 'Upload a photo':
    image_upload = st.file_uploader('upload file', type = ['jpg', 'png', 'jpeg'])
    if image_upload != None:
        image_np = np.asarray(bytearray(image_upload.read()),dtype = np.uint8)
        img = cv2.imdecode(image_np,1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        index = np.argmax(prediction[0])
        money = money_type[index]
        st.image(image_upload)
        st.write('This is:', money, 'VND')
                
