import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from twilio.rest import Client
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Twilio credentials from .env ───────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUM    = os.getenv("TWILIO_FROM_NUM")   # e.g. whatsapp:+14155238886
TWILIO_TO_NUM      = os.getenv("TWILIO_TO_NUM")     # e.g. whatsapp:+919xxxxxxxxx

# ── PAGE CONFIG ────────────────────────────────
st.set_page_config(
    page_title="FarmGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    header { visibility: hidden !important; }
    .stApp { background-color: #21252b; color: #a9b2be; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    [data-testid="stSidebar"] { background-color: #2c3e44 !important; border-right: none; }
    [data-testid="stSidebar"] * { color: #f0f4f8 !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem !important; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 600 !important; }
    .main-title { display: flex; align-items: center; gap: 15px; font-size: 2rem; font-weight: 700; color: #e2e8f0; margin-top: 0; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #363b44; }
    .custom-panel { background-color: #262b32; border: 1px solid #3d4450; border-radius: 12px; padding: 15px 20px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .panel-header { display: flex; align-items: center; gap: 10px; font-size: 1.2rem; font-weight: 600; color: #ffffff; margin-bottom: 10px; }
    .stSelectbox > div > div { background-color: #212e33 !important; border: 1px solid #4a5d63 !important; color: white !important; border-radius: 6px; }
    .stSlider > div > div > div > div { background-color: #5da3a5 !important; }
    div.stButton > button[kind="primary"] { background-color: #649fa1 !important; color: white !important; border: none !important; border-radius: 20px !important; padding: 8px 24px !important; font-weight: 600 !important; width: 100%; transition: all 0.3s; margin-bottom: 5px; }
    div.stButton > button[kind="primary"]:hover { background-color: #528688 !important; box-shadow: 0 0 10px rgba(100,159,161,0.4); }
    div.stButton > button[kind="secondary"] { background-color: #c45c5c !important; color: white !important; border: none !important; border-radius: 20px !important; padding: 8px 24px !important; font-weight: 600 !important; width: 100%; transition: all 0.3s; }
    div.stButton > button[kind="secondary"]:hover { background-color: #a84b4b !important; box-shadow: 0 0 10px rgba(196,92,92,0.4); }
    .overview-title { font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 10px; }
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 15px; }
    .stat-box { background-color: #2a2e33; border-radius: 8px; padding: 12px 12px; }
    .stat-number { font-size: 22px; font-weight: bold; color: #ffffff; line-height: 1; margin-bottom: 4px; }
    .stat-number.red { color: #ef4444; }
    .stat-label { font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }
    .log-divider { border-top: 1px solid #3d4450; margin: 0 -20px 10px -20px; }
    .log-section-title { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .log-container { height: 350px; overflow-y: auto; padding-right: 5px; }
    .log-item { display: flex; align-items: flex-start; margin-bottom: 12px; }
    .icon-circle { width: 22px; height: 22px; background-color: #ffffff; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-right: 12px; margin-top: 2px; }
    .icon-triangle-red { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 9px solid #b91c1c; margin-bottom: 2px; }
    .icon-triangle-yellow { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 9px solid #d97706; margin-bottom: 2px; }
    .icon-square-green { width: 9px; height: 9px; background-color: #16a34a; border-radius: 1px; }
    .log-text { flex-grow: 1; line-height: 1.3; }
    .log-title { color: #ffffff; font-size: 14px; font-weight: 600; }
    .log-subtitle { color: #9ca3af; font-size: 12px; margin-top: 2px; }
    .log-time { color: #8b9bb4; font-size: 11px; flex-shrink: 0; margin-left: 10px; margin-top: 2px; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #4a5568; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── INIT ───────────────────────────────────────
os.makedirs("detections", exist_ok=True)
animal_classes = ["buffalo", "elephant", "rhino", "zebra"]

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ── TWILIO WHATSAPP ALERT ──────────────────────
def send_whatsapp_alert(animal):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUM, TWILIO_TO_NUM]):
        return False
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"⚠️ FarmGuard ALERT: {animal.upper()} detected near the perimeter! [{datetime.now().strftime('%H:%M:%S')}]",
            from_=TWILIO_FROM_NUM,
            to=TWILIO_TO_NUM
        )
        return True
    except Exception as e:
        st.sidebar.error(f"Twilio error: {e}")
        return False

# ── CREDENTIAL CHECK ───────────────────────────
def check_credentials():
    missing = []
    if not TWILIO_ACCOUNT_SID: missing.append("TWILIO_ACCOUNT_SID")
    if not TWILIO_AUTH_TOKEN:  missing.append("TWILIO_AUTH_TOKEN")
    if not TWILIO_FROM_NUM:    missing.append("TWILIO_FROM_NUM")
    if not TWILIO_TO_NUM:      missing.append("TWILIO_TO_NUM")
    return missing

# ── SIDEBAR ────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    video_source = st.selectbox("Video Source", ["zebr.mp4", "Webcam (0)", "elep.mp4", "bufflo.mp4"])
    confidence   = st.slider("Detection Confidence", 0.1, 1.0, 0.40, 0.05)
    night_vision = st.toggle("🌙 Enable Night Vision Enhancement", value=False)

    st.markdown("<hr style='border-color: #4a5d63; margin: 15px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📞 Alert Configuration")
    alert_interval = st.number_input("Alert Interval (seconds)", min_value=10, value=60)

    missing = check_credentials()
    if missing:
        st.error("Missing .env keys:\n" + "\n".join(f"• {k}" for k in missing))
    else:
        st.success("✅ Twilio credentials loaded")

    start_button = st.button("▶ Start Monitoring", type="primary")
    stop_button  = st.button("⏹ Stop Monitoring",  type="secondary")

# ── MAIN TITLE ─────────────────────────────────
st.markdown('<div class="main-title">🛡️ FarmGuard Wildlife Detection System</div>',
            unsafe_allow_html=True)

col_main, col_side = st.columns([2.2, 1])

# ── SESSION STATE ──────────────────────────────
if 'logs' not in st.session_state:
    st.session_state['logs'] = [{
        "type": "info",
        "title": "System initialized",
        "sub": "Ready for monitoring",
        "time": datetime.now().strftime('%H:%M:%S')
    }]
    st.session_state['last_alert_time'] = 0
    st.session_state['is_running']      = False
    st.session_state['total_detected']  = 0
    st.session_state['alerts_sent']     = 0

if start_button: st.session_state['is_running'] = True
if stop_button:  st.session_state['is_running'] = False

# ── LOG RENDERING ──────────────────────────────
def get_icon_html(log_type):
    if log_type == "danger":
        return '<div class="icon-circle"><div class="icon-triangle-red"></div></div>'
    elif log_type == "warning":
        return '<div class="icon-circle"><div class="icon-triangle-yellow"></div></div>'
    return '<div class="icon-circle"><div class="icon-square-green"></div></div>'

with col_side:
    log_panel_placeholder = st.empty()

def render_logs():
    html = f"""<div class="custom-panel">
<div class="overview-title">Session overview</div>
<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-number">{st.session_state['total_detected']}</div>
        <div class="stat-label">DETECTED</div>
    </div>
    <div class="stat-box">
        <div class="stat-number red">{st.session_state['alerts_sent']}</div>
        <div class="stat-label">ALERTS SENT</div>
    </div>
</div>
<div class="log-divider"></div>
<div class="log-section-title">DETECTION LOG</div>
<div class="log-container">"""

    for log in st.session_state['logs']:
        icon = get_icon_html(log['type'])
        html += f"""
<div class="log-item">
    {icon}
    <div class="log-text">
        <div class="log-title">{log['title']}</div>
        <div class="log-subtitle">{log['sub']}</div>
    </div>
    <div class="log-time">{log['time']}</div>
</div>"""

    html += '</div></div>'
    log_panel_placeholder.markdown(html, unsafe_allow_html=True)

render_logs()

# ── VIDEO PANEL ────────────────────────────────
with col_main:
    st.markdown('<div class="custom-panel" style="padding-bottom: 5px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">📷 Live Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

    if not st.session_state['is_running']:
        video_placeholder.markdown("""
        <div style="width:100%;height:450px;background-color:#1a1e23;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;
                    color:#4a5568;font-size:1.2rem;border:1px solid #2d343f;">
            Camera Offline — Press 'Start Monitoring'
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# # ── DETECTION LOOP ─────────────────────────────
# if st.session_state['is_running']:

#     if len(st.session_state['logs']) == 1 and "initialized" in st.session_state['logs'][0]['title']:
#         st.session_state['logs'].insert(0, {
#             "type": "info",
#             "title": "Monitoring started",
#             "sub": "YOLOv8 initialized",
#             "time": datetime.now().strftime('%H:%M:%S')
#         })
#         render_logs()

#     model  = load_model("best.pt")
#     source = 0 if video_source == "Webcam (0)" else video_source
#     cap    = cv2.VideoCapture(source)

#     while cap.isOpened() and st.session_state['is_running']:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Video stream ended.")
#             break

#         if night_vision:
#             lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             cl    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
#             frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

#         results          = model(frame, conf=confidence, verbose=False)
#         detected_animals = set()

#         for r in results:
#             for box in r.boxes:
#                 label = model.names[int(box.cls[0])]
#                 conf  = float(box.conf[0])
#                 if label in animal_classes:
#                     detected_animals.add((label, conf))

#         annotated = frame.copy()
#         for r in results:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = model.names[int(box.cls[0])]
#                 conf  = float(box.conf[0])
#                 if label in animal_classes:
#                     cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 190, 130), 2)
#                     text = f"{label.capitalize()} {int(conf*100)}%"
#                     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#                     cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), (180, 190, 130), -1)
#                     cv2.putText(annotated, text, (x1 + 4, y1 - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#         video_placeholder.image(
#             cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
#             channels="RGB",
#             use_container_width=True
#         )

#         # ── ALERT LOGIC ────────────────────────
#         current_time = time.time()
#         if detected_animals and (current_time - st.session_state['last_alert_time'] >= alert_interval):
#             top_animal, top_conf = max(detected_animals, key=lambda x: x[1])
#             time_str = datetime.now().strftime('%H:%M:%S')

#             st.session_state['total_detected'] += 1
#             log_type = "warning" if top_animal in ["zebra", "buffalo"] else "danger"

#             st.session_state['logs'].insert(0, {
#                 "type": log_type,
#                 "title": f"{top_animal.capitalize()} detected",
#                 "sub": f"Confidence {int(top_conf*100)}% · {video_source.split('.')[0].capitalize()}",
#                 "time": time_str
#             })

#             alert_ok = send_whatsapp_alert(top_animal)
#             if alert_ok:
#                 st.session_state['alerts_sent'] += 1
#                 masked = TWILIO_TO_NUM[-4:].rjust(len(TWILIO_TO_NUM), '*')
#                 st.session_state['logs'].insert(0, {
#                     "type": "info",
#                     "title": "WhatsApp alert sent",
#                     "sub": f"Delivered to {masked}",
#                     "time": time_str
#                 })
#             else:
#                 st.session_state['logs'].insert(0, {
#                     "type": "warning",
#                     "title": "Alert failed",
#                     "sub": "Check Twilio credentials in .env",
#                     "time": time_str
#                 })

#             st.session_state['last_alert_time'] = current_time

#             if len(st.session_state['logs']) > 20:
#                 st.session_state['logs'] = st.session_state['logs'][:20]

#             render_logs()

#     cap.release()
#     st.session_state['is_running'] = False
#     st.session_state['logs'].insert(0, {
#         "type": "info",
#         "title": "Monitoring stopped",
#         "sub": "System offline",
#         "time": datetime.now().strftime('%H:%M:%S')
#     })
#     st.rerun()



# ── DETECTION LOOP ─────────────────────────────
# ── DETECTION LOOP ─────────────────────────────
if st.session_state['is_running']:

    model  = load_model("best.pt")
    source = 0 if video_source == "Webcam (0)" else video_source
    cap    = cv2.VideoCapture(source)

    frame_window = video_placeholder

    for _ in range(100000):   # simulate continuous stream

        if not st.session_state['is_running']:
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("Video stream ended.")
            break

        if night_vision:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        results          = model(frame, conf=confidence, verbose=False)
        detected_animals = set()

        annotated = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                conf  = float(box.conf[0])

                if label in animal_classes:
                    detected_animals.add((label, conf))

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (180,190,130), 2)

        frame_window.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        # ALERT LOGIC
        current_time = time.time()
        if detected_animals and (current_time - st.session_state['last_alert_time'] >= alert_interval):
            top_animal, _ = max(detected_animals, key=lambda x: x[1])
            st.session_state['last_alert_time'] = current_time
            send_whatsapp_alert(top_animal)

        time.sleep(0.03)

    cap.release()
      
