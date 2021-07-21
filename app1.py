import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image

mp_drawings=mp.solutions.drawing_utils
mp_face_mesh=mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

st.title("IMAGE SEGMENTATION AND FACE DETECTION")
st.subheader("CREATED BY SHIVA SAI")
selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("ABOUT", "FACE DETECTION", "SELFIE SEGMENTATION")
)
if selectbox=="ABOUT":
    st.write("This is a part of Lets Upgrade tutorial helped me to build an app")
    st.write("This app performs various rendering on images try by uploading an image")
elif selectbox=="FACE DETECTION":
    image_path=st.sidebar.file_uploader("Upload an Image")
    if image_path is not None:
        image=np.array(Image.open(image_path))
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            results=face_detection.process(image)
            annotated_image = image.copy()
            for detection in results.detections:
                print('Nose tip:')
                print(mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawings.draw_detection(annotated_image, detection)
            st.image(annotated_image)
elif selectbox=="SELFIE SEGMENTATION":
    button=st.sidebar.radio("Color",("None","red","green","IRON MAN BACKGROUND","AVENGERS BACKGROUND"))
    image_path=st.sidebar.file_uploader("Upload an Image")
    if button=="None":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            st.image(image)
            st.write("Choose any method to segment your image by chossing on respective button")
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")
    elif button=="red":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            r,g,b=cv2.split(image)
            zeros=np.zeros(image.shape[:2],dtype="uint8")
            st.image(cv2.merge([r,zeros,zeros]))
            st.write("Given Image's background is changed to red")
    elif button=="green":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            r,g,b=cv2.split(image)
            zeros=np.zeros(image.shape[:2],dtype="uint8")
            st.image(cv2.merge([zeros,g,zeros]))
            st.write("Given Image's background is changed to red")
    elif button=="IRON MAN BACKGROUND":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            ironman=cv2.imread("iron.jpg")
            ironman=cv2.resize(ironman,(image.shape[1],image.shape[0]))
            ironman=cv2.cvtColor(ironman,cv2.COLOR_BGR2RGB)
            image_intensity=st.sidebar.select_slider("MAIN IMAGE INTENSITY",options=[0.5,0.6,0.7,0.8,0.9,1])
            background=st.sidebar.select_slider("Background Image Intensity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            alpha=st.sidebar.select_slider("opcaity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            blended=cv2.addWeighted(image,image_intensity,ironman,background,alpha)
            st.image(blended)
    elif button=="AVENGERS BACKGROUND":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            ironman=cv2.imread("avengers.jpg")
            ironman=cv2.resize(ironman,(image.shape[1],image.shape[0]))
            ironman=cv2.cvtColor(ironman,cv2.COLOR_BGR2RGB)
            image_intensity=st.sidebar.select_slider("MAIN IMAGE INTENSITY",options=[0.5,0.6,0.7,0.8,0.9,1])
            background=st.sidebar.select_slider("Background Image Intensity",options=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            alpha=st.sidebar.select_slider("opcaity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            blended=cv2.addWeighted(image,image_intensity,ironman,background,alpha)
            st.image(blended)