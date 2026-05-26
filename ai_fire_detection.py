from ultralytics import YOLO
import cv2
import requests
import datetime
import os
import time
import csv
import pygame

# Telegram Bot Configuration

BOT_TOKEN = "8901728129:AAHm41XTr-klQ7iS_YqoZUQPWaLSC55_z7c"
CHAT_ID = "794630841"

# Load AI Model

print("Loading AI model...")

model = YOLO("best.pt")

print("Model loaded successfully ✅")

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("assests/siren.wav")

# Create screenshots directory

os.makedirs("screenshots", exist_ok=True)

# Create CSV log file.

log_file = "fire_logs.csv"

if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Confidence", "Screenshot"])

# Access webcam

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Cannot access webcam ❌")
    exit()

print("🔥 AI Fire Detection Started")
print("Press Q to Exit")

# Cooldown mechanism to prevent alert spamming

last_alert_time = 0
cooldown_seconds = 15

# Main loop

while True:

    ret, frame = video.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Run detection on the frame

    results = model(frame, device=0)

    annotated_frame = results[0].plot()

   # Extract bounding boxes and confidence scores

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        current_time = time.time()

        # Prevent spam alerts
        if current_time - last_alert_time > cooldown_seconds:

            for box in boxes:

                confidence = float(box.conf[0])

                # Confidence threshold
                if confidence > 0.50:

                    alarm_sound.play() # Play alarm sound

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    screenshot_path = f"screenshots/fire_{timestamp}.jpg"

                    # Save screenshot
                    cv2.imwrite(screenshot_path, frame)

                    print(f"🔥 FIRE DETECTED! Confidence: {confidence:.2f}")

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

📸 Screenshot Saved
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
                        response = requests.post(telegram_url, data=payload)

                        if response.status_code == 200:
                            print("✅ Telegram Alert Sent")
                        else:
                            print("❌ Telegram Error")

                    except Exception as e:
                        print("Telegram Exception:", e)

                    # Update cooldown timer
                    last_alert_time = current_time

    # Display the annotated frame

    cv2.imshow("🔥 AI Fire Detection System", annotated_frame)

    # Exit on 'Q' key press

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing system...")
        break

# cleanup

video.release()
cv2.destroyAllWindows()

print("✅ System Closed Successfully")