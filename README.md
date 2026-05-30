#  AI Fire Detection & Surveillance System

A real-time AI-powered fire detection and surveillance system built using YOLOv8, OpenCV, Streamlit, and GPU-accelerated PyTorch.

The system continuously monitors live webcam feeds, detects fire in real time, triggers emergency alerts, plays siren alarms, captures screenshots, logs incidents, and sends Telegram notifications automatically.

---

#  Features

-  Real-time fire detection using YOLOv8
-  Live webcam surveillance
-  GPU acceleration with NVIDIA CUDA
-  Telegram emergency alerts
-  Siren alarm system
-  Automatic screenshot capture
-  CSV-based incident logging
-  Interactive Streamlit dashboard
-  Detection analytics and logs

---

#  Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- PyTorch
- CUDA
- Pandas
- Pygame
- Telegram Bot API

---

#  System Architecture

```text
Webcam Feed
     ↓
YOLOv8 Fire Detection Model
     ↓
Fire Detection Event
     ↓
├── Telegram Alert
├── Siren Alarm
├── Screenshot Capture
├── CSV Logging
└── Streamlit Dashboard
```

---

#  GPU Acceleration

The project uses CUDA-enabled PyTorch for GPU inference acceleration on NVIDIA RTX GPUs.

Example GPU configuration:

- NVIDIA RTX 2050
- CUDA 12.5
- PyTorch CUDA Build (cu124)

---

#  Project Structure

```text
Fire_detection/
│
├── assests/
│   └── siren.wav
│
├── notebooks/
│   └── fire_detection_v1.ipynb
│
├── screenshots/
│
├── app.py
├── ai_fire_detection.py
├── gpu_check.py
├── best.pt
├── README.md
├── requirements.txt
└── .gitignore
```

---

#  Installation

## Clone Repository

```bash
git clone YOUR_GITHUB_REPO_LINK
```

## Navigate to Project

---bash
cd Fire_detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Run Application

---bash
streamlit run app.py
```

---

#  Screenshots

Dashboard Image
assests/fire_detect(Dashboard).png



#  Future Improvements

- Cloud deployment
- Multi-camera support
- Smoke detection
- Mobile notifications
- Industrial CCTV integration
- Edge AI optimization

---

## Final Year project in in my 
notebooks/fire_detection_v1.ipynb 


#  Author

Aditya Motghare

AI & Computer Vision Enthusiast