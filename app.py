import streamlit as st
import pandas as pd
import os
from PIL import Image
import cv2
from ultralytics import YOLO
import requests
import datetime
import time
import csv
import pygame

# Page configuration

st.set_page_config(
    page_title="AI Fire Detection Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Telegram

BOT_TOKEN = "8901728129:AAHm41XTr-klQ7iS_YqoZUQPWaLSC55_z7c"
CHAT_ID = "794630841"

# Load AI Model

model = YOLO("best.pt")

# Initialize Pygame for alarm sound

pygame.mixer.init()

alarm_sound = pygame.mixer.Sound("assests/siren.wav")

# Create screenshots directory

os.makedirs("screenshots", exist_ok=True)

# Create CSV log file

log_file = "fire_logs.csv"

if not os.path.exists(log_file):

    with open(log_file, mode='w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "Confidence",
            "Screenshot"
        ])

# Title and description

st.title("🔥 AI Fire Detection Dashboard")

st.markdown("---")

# Live Camera Feed

st.subheader("📹 Live AI Surveillance Feed")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

last_alert_time = 0
cooldown_seconds = 15
frame_count = 0

while run:

    success, frame = camera.read()

    if not success:

        st.error("Failed to access webcam")
        break

    frame_count += 1

    # Skip alternate frames AFTER reading frame
    if frame_count % 2 != 0:
        continue

    # Resize frame for faster inference
    frame = cv2.resize(frame, (640, 360))


    # Yolo Inference

    results = model(frame, device=0)

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        current_time = time.time()

        if current_time - last_alert_time > cooldown_seconds:

            for box in boxes:

                confidence = float(box.conf[0])

                if confidence > 0.55:

                    print("Fire Deteted with confidence:", confidence)

                   # Play alarm sound

                    alarm_sound.play()

                    timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )

                    screenshot_path = f"screenshots/fire_{timestamp}.jpg"

                    cv2.imwrite(
                        screenshot_path,
                        frame
                    )

                    # Save log to CSV

                    with open(log_file, mode='a', newline='') as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            timestamp,
                            f"{confidence:.2f}",
                            screenshot_path
                        ])

                    # Telegram Alert

                    message = f'''
🔥 FIRE DETECTED!

⏰ Time: {timestamp}

🎯 Confidence: {confidence:.2f}

⚠ Emergency Attention Required
'''

                    telegram_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

                    payload = {
                        "chat_id": CHAT_ID,
                        "text": message
                    }

                    try:

                        response = requests.post(
                            telegram_url,
                            data=payload
                        )

                        print(response.text)

                        st.success("✅ Telegram Alert Sent")

                    except Exception as e:

                        st.error(f"Telegram Error: {e}")

                    last_alert_time = current_time

    # Display the annotated frame

    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    FRAME_WINDOW.image(annotated_frame)

camera.release()

st.markdown("---")

# System Status

st.subheader("🚨 System Status")

st.success("AI Fire Detection System Running")

# Logs

if os.path.exists(log_file):

    df = pd.read_csv(log_file)

    st.subheader("📋 Detection Logs")

    st.dataframe(df)

    st.subheader("🔥 Total Fire Detections")

    st.metric(
        label="Detections",
        value=len(df)
    )

# Screenshot Gallery

screenshot_folder = "screenshots"

st.subheader("📸 Latest Fire Screenshots")

if os.path.exists(screenshot_folder):

    images = os.listdir(screenshot_folder)

    images = sorted(images, reverse=True)

    if len(images) > 0:

        latest_image = images[0]

        image_path = os.path.join(
            screenshot_folder,
            latest_image
        )

        image = Image.open(image_path)

        st.image(
            image,
            caption=latest_image,
            width="stretch"
        )