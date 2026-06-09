import os
import streamlit as st
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision import Detections
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

model_path = 'yolov12n_vehiclebest.pt'
model = YOLO(model_path)
classes = ['bus', 'car', 'van']

import json
import datetime

st.set_page_config(page_title="Vehicle Detection", layout="wide")
st.title("Vehicle Detection")

## ------------------------------    Sidebar      -----------------------------------------------------------------
st.sidebar.header("Model Configurations")
model_type = st.sidebar.radio("Task", ["Detection"])

## -----------------------------------------    Vehicle Detection       ---------------------------------------------
st.markdown("Upload an image to detect vehicles")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def reannotate_image(image_rgb, detections):
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    labels = [f"{classes[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)]
    
    annotated_image = image_rgb.copy()
    if len(detections) > 0:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(image_rgb, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate and display image
    annotated_image = reannotate_image(image_rgb, detections)
    st.image(annotated_image, caption="Image for Vehicle Detections", use_container_width=True)
    
    # Count vehicles
    car_detections = detections[detections.class_id == classes.index('car')] if len(detections) > 0 else sv.Detections.empty()
    van_detections = detections[detections.class_id == classes.index('van')] if len(detections) > 0 else sv.Detections.empty()
    bus_detections = detections[detections.class_id == classes.index('bus')] if len(detections) > 0 else sv.Detections.empty()
        
    car_qty = len(car_detections)
    van_qty = len(van_detections)
    bus_qty = len(bus_detections)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🚗 Cars", car_qty)
    with col2: st.metric("🚐 Vans", van_qty)
    with col3: st.metric("🚌 Buses", bus_qty)
    
    ## ----------------------   OCR   -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("License Plate Recognition")
    
    def ai_ocr_carplate(image_np):
        try:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            b64_image = base64.b64encode(buffer).decode("utf-8")
            
            response = client.chat.completions.create(model='gpt-4o-mini',
                messages=[{"role": "user",
                        "content": [{"type": "text", "text": "Extract every vehicle license plate number detected as a bus, van, or car from this image. Return only the plate numbers as a comma-separated list. If no plates are visible, return 'None'."},
                            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}],)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    if st.button("🔍 Extract License Plates with AI"):
        with st.spinner("Analyzing image with GPT-4o..."):
            plate_result = ai_ocr_carplate(image_rgb)
            st.success("Analysis Complete!")
            st.subheader("Detected License Plates:")
            st.write(plate_result)

else:
    st.info("📤 Please upload an image to get started.")
