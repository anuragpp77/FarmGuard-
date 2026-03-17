# 🛡️ FarmGuard — Wildlife Intrusion Detection System

> A Streamlit-based wildlife monitoring dashboard that uses a fine-tuned YOLOv8 model to detect farm-threatening animals in real time and fires WhatsApp alerts via Twilio when a threat is detected.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple) ![Twilio](https://img.shields.io/badge/Alerts-Twilio%20WhatsApp-green)

---

## 📌 Overview

FarmGuard is a real-time wildlife detection and alert system designed to protect farms from animal intrusions. It uses a fine-tuned **YOLOv8** object detection model to identify dangerous or crop-damaging animals from live video feeds (webcam or pre-recorded), and instantly dispatches **WhatsApp alerts** via Twilio when a threat is detected.

The dashboard is built with **Streamlit** and features a clean dark-themed UI with a live annotated video feed, detection logs, session statistics, and optional night-vision enhancement.

---

## ✨ Features

- 🎥 **Live Video Monitoring** — supports webcam input or video file playback
- 🦣 **Multi-Animal Detection** — identifies elephants, buffalos, rhinos, and zebras using YOLOv8
- 📱 **WhatsApp Alerts** — instant notifications via Twilio WhatsApp API with configurable alert intervals
- 🌙 **Night Vision Enhancement** — CLAHE-based low-light image enhancement for nighttime monitoring
- 📊 **Session Dashboard** — real-time detection count, alert count, and timestamped detection log
- 🎛️ **Adjustable Confidence Threshold** — tune detection sensitivity via a slider
- 🖤 **Dark-themed UI** — custom CSS for a professional monitoring console aesthetic

---

## 🐾 Detectable Animals

| Animal     | Risk Level |
|------------|------------|
| 🐘 Elephant | High       |
| 🦏 Rhino    | High       |
| 🐃 Buffalo  | Medium     |
| 🦓 Zebra    | Medium     |

---

## 🗂️ Project Structure

```
farmguard/
├── app.py              # Main Streamlit application
├── best.pt             # Fine-tuned YOLOv8 model weights (not included — see below)
├── zebr.mp4            # Sample zebra video (not included — add your own)
├── elep.mp4            # Sample elephant video (not included — add your own)
├── bufflo.mp4          # Sample buffalo video (not included — add your own)
├── detections/         # Auto-created folder for saved detection frames
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/farmguard.git
cd farmguard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If deploying on a headless server (no display), replace `opencv-python` with `opencv-python-headless` in `requirements.txt`.

### 3. Add required assets

- Place your fine-tuned YOLOv8 weights file as `best.pt` in the project root.
- Optionally add sample video files (`zebr.mp4`, `elep.mp4`, `bufflo.mp4`) for testing.

---

## 🔑 Configuration

Before running, open `app.py` and update the Twilio credentials at the top of the file:

```python
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN  = 'your_auth_token'
TWILIO_FROM_NUM    = 'whatsapp:+14155238886'   # Twilio sandbox number
TWILIO_TO_NUM      = 'whatsapp:+91XXXXXXXXXX'  # Your WhatsApp number
```

> You can get a free Twilio WhatsApp sandbox at [twilio.com/console](https://www.twilio.com/console).

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

---

## 🖥️ Usage

1. **Select a video source** from the sidebar (webcam or a video file).
2. **Adjust the confidence threshold** to control detection sensitivity.
3. **Toggle Night Vision** if monitoring in low-light conditions.
4. **Set the alert interval** to avoid notification flooding (default: 60 seconds).
5. Click **▶ Start Monitoring** to begin.
6. Click **⏹ Stop Monitoring** to end the session.

Detection events appear in the **live log panel** on the right, and WhatsApp alerts are dispatched automatically.

---

## 📦 Requirements

```
streamlit>=1.35.0
opencv-python>=4.9.0
numpy>=1.26.0
ultralytics>=8.2.0
twilio>=9.2.0
```

---

## 🛠️ Tech Stack

| Component        | Technology              |
|------------------|-------------------------|
| UI Framework     | Streamlit               |
| Object Detection | YOLOv8 (Ultralytics)    |
| Video Processing | OpenCV                  |
| Alerts           | Twilio WhatsApp API     |
| Image Processing | NumPy, OpenCV CLAHE     |

---



---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Twilio](https://www.twilio.com/)
- [Streamlit](https://streamlit.io/)
