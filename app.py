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
from dotenv import load_dotenv
import torch

# Load environment variables

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# page configuration

st.set_page_config(
    page_title="AI Fire Detection Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Page title

st.title("MY AI Fire Detection Dashboard")

st.markdown("---")

# GPU Check

st.sidebar.header(" System Information")

if torch.cuda.is_available():

    st.sidebar.success("GPU Active ")

    st.sidebar.write(torch.cuda.get_device_name(0))

else:

    st.sidebar.error("Running on CPU NOT GPU")

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

# Access webcam

st.subheader("MY Live AI Surveillance Feed")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

# Alert cooldown mechanism

last_alert_time = 0
cooldown_seconds = 15

# Frame counter for performance optimization

frame_count = 0

# Main loop

while run:

    success, frame = camera.read()

    if not success:

        st.error(" Failed to access webcam")
        break

    frame_count += 1

    # Skip alternate frames for performance

    if frame_count % 2 != 0:
        continue

    # Resize frame for faster inference

    frame = cv2.resize(frame, (640, 360))

    # YOLOv8 inference

    results = model(frame, device=0)

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    # Check for detections and handle alerts

    if boxes is not None and len(boxes) > 0:

        current_time = time.time()

        if current_time - last_alert_time > cooldown_seconds:

            for box in boxes:

                confidence = float(box.conf[0])

                # Confidence threshold

                if confidence > 0.55:

                    print(f"YESS! FIRE DETECTED | Confidence: {confidence:.2f}")

                    # Play alarm sound

                    try:

                        alarm_sound.play()

                    except Exception as e:

                        print("Alarm Error:", e)

                    # -----------------------------------
                    # TIMESTAMP
                    # -----------------------------------

                    timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )

                    # Save screenshot

                    screenshot_path = f"screenshots/fire_{timestamp}.jpg"

                    cv2.imwrite(
                        screenshot_path,
                        frame
                    )

                    # Log detection to CSV

                    with open(log_file, mode='a', newline='') as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            timestamp,
                            f"{confidence:.2f}",
                            screenshot_path
                        ])

                    # Send Telegram alert

                    message = f"""
🔥 FIRE DETECTED!

⏰ Time: {timestamp}

🎯 Confidence: {confidence:.2f}

⚠ Emergency Attention Required
"""

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

                    # -----------------------------------
                    # UPDATE COOLDOWN TIMER
                    # -----------------------------------

                    last_alert_time = current_time

    # -----------------------------------
    # DISPLAY FRAME
    # -----------------------------------

    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    FRAME_WINDOW.image(
        annotated_frame,
        channels="RGB",
        width="stretch"
    )

# Exit on 'Q' key press

camera.release()

# -----------------------------------
# SYSTEM STATUS
# -----------------------------------

st.markdown("---")

st.subheader("MY System Status")

st.success("AI Fire Detection System Running")

# -----------------------------------
# SHOW LOGS
# -----------------------------------

if os.path.exists(log_file):

    df = pd.read_csv(log_file)

    st.subheader(" Detection Logs")

    st.dataframe(
        df,
        width="stretch"
    )

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