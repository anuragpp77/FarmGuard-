import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import time
import pandas as pd
from datetime import datetime, timedelta
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FarmGuard · Animal Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:      #0d1a0f;
    --bg-card:      #131f14;
    --bg-raised:    #1a2b1c;
    --border:       #243326;
    --accent-lime:  #a3e635;
    --accent-amber: #f59e0b;
    --accent-red:   #ef4444;
    --text-primary: #e8f0e9;
    --text-muted:   #6b8f70;
    --text-dim:     #3d5c42;
    --serif:        'DM Serif Display', Georgia, serif;
    --sans:         'DM Sans', system-ui, sans-serif;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: var(--sans) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }

.hero {
    display: flex; align-items: flex-end; gap: 1.5rem;
    padding: 2.8rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-icon { font-size: 3.6rem; line-height: 1; filter: drop-shadow(0 0 18px rgba(163,230,53,.35)); }
.hero-title { font-family: var(--serif) !important; font-size: 2.8rem !important;
    color: var(--text-primary) !important; letter-spacing: -0.5px; line-height: 1; margin: 0 !important; }
.hero-sub { font-size: 0.85rem; color: var(--text-muted); letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 0.35rem; }
.live-badge {
    margin-left: auto; display: flex; align-items: center; gap: 0.5rem;
    background: rgba(163,230,53,.1); border: 1px solid rgba(163,230,53,.3);
    border-radius: 999px; padding: 0.35rem 0.9rem;
    font-size: 0.78rem; font-weight: 600; color: var(--accent-lime); letter-spacing: 0.08em;
}
.live-dot {
    width: 7px; height: 7px; background: var(--accent-lime); border-radius: 50%;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.75)} }

.metric-row { display:flex; gap:1rem; margin-bottom:1.75rem; }
.metric-card {
    flex:1; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.25rem 1.5rem; position: relative; overflow: hidden;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, var(--accent-lime), transparent); }
.metric-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), transparent); }
.metric-card.red::before   { background: linear-gradient(90deg, var(--accent-red), transparent); }
.metric-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:var(--text-muted); margin-bottom:.4rem; }
.metric-val   { font-family:var(--serif); font-size:2.2rem; color:var(--text-primary); line-height:1; }
.metric-delta { font-size:.75rem; color:var(--text-muted); margin-top:.25rem; }

.section-title {
    font-family: var(--serif); font-size: 1.35rem; color: var(--text-primary);
    margin-bottom: 1rem; display: flex; align-items: center; gap: .6rem;
}
.section-line { flex:1; height:1px; background: linear-gradient(90deg, var(--border), transparent); }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important; border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important; transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(163,230,53,.45) !important; }
[data-testid="stFileUploader"] label { color: var(--text-muted) !important; }

.detection-result {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: .75rem;
}
.det-header { display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem; }
.det-badge { display:inline-block; padding:.2rem .65rem; border-radius:999px;
    font-size:.72rem; font-weight:600; letter-spacing:.06em; }
.badge-danger { background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.3); }
.badge-warn   { background:rgba(245,158,11,.15); color:#fbbf24; border:1px solid rgba(245,158,11,.3); }
.badge-safe   { background:rgba(163,230,53,.12); color:var(--accent-lime); border:1px solid rgba(163,230,53,.3); }
.det-name     { font-weight:600; font-size:1rem; color:var(--text-primary); }
.conf-bar-bg  { background:var(--bg-raised); border-radius:999px; height:5px; overflow:hidden; margin-top:.45rem; }
.conf-bar     { height:100%; border-radius:999px; background: linear-gradient(90deg, var(--accent-lime), #65a30d); }

.alert-pill {
    display:flex; align-items:center; gap:.7rem;
    background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.25);
    border-radius:10px; padding:.9rem 1.2rem; margin-bottom:.6rem;
    font-size:.85rem; color:#fca5a5;
}

.sidebar-brand { font-family: var(--serif); font-size:1.25rem; color: var(--accent-lime);
    padding:1rem 0 .75rem; border-bottom:1px solid var(--border); margin-bottom:1rem; }
.sidebar-section { font-size:.7rem; text-transform:uppercase;
    letter-spacing:.12em; color:var(--text-dim); margin:.9rem 0 .4rem; }

label { color: var(--text-muted) !important; font-size:.82rem !important; }
.stButton button {
    background: var(--accent-lime) !important; color: #0d1a0f !important;
    font-weight: 700 !important; border: none !important; border-radius: 8px !important;
    padding: .55rem 1.4rem !important; font-size: .88rem !important;
    letter-spacing: .03em !important; transition: opacity .15s !important;
}
.stButton button:hover { opacity:.85 !important; }

.footer {
    margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
    text-align: center; font-size: .75rem; color: var(--text-dim); letter-spacing: .05em;
}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ANIMALS = {
    "Wild Boar":   {"threat": "danger", "emoji": "🐗", "color": (239, 68,  68)},
    "Deer":        {"threat": "warn",   "emoji": "🦌", "color": (245,158,  11)},
    "Elephant":    {"threat": "danger", "emoji": "🐘", "color": (239, 68,  68)},
    "Fox":         {"threat": "warn",   "emoji": "🦊", "color": (245,158,  11)},
    "Rabbit":      {"threat": "safe",   "emoji": "🐰", "color": (163,230,  53)},
    "Monkey":      {"threat": "warn",   "emoji": "🐒", "color": (245,158,  11)},
    "Leopard":     {"threat": "danger", "emoji": "🐆", "color": (239, 68,  68)},
    "Crow":        {"threat": "safe",   "emoji": "🐦", "color": (163,230,  53)},
    "Rat":         {"threat": "warn",   "emoji": "🐀", "color": (245,158,  11)},
    "Porcupine":   {"threat": "warn",   "emoji": "🦔", "color": (245,158,  11)},
}

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌾 FarmGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Detection Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.30, 0.95, 0.55, 0.05)
    iou_threshold  = st.slider("IoU Threshold (NMS)",  0.30, 0.80, 0.45, 0.05)
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Detector Model",
        ["YOLOv8-FarmGuard-L", "YOLOv8-FarmGuard-S", "RT-DETR-v2", "YOLOv5-Agri"])
    device = st.selectbox("Inference Device", ["CPU", "CUDA GPU", "MPS (Apple Silicon)"])
    st.markdown('<div class="sidebar-section">Alerts</div>', unsafe_allow_html=True)
    alert_danger = st.toggle("Alert on Dangerous Animals", value=True)
    alert_warn   = st.toggle("Alert on Warning Animals",   value=True)
    sms_notify   = st.toggle("SMS / Email Notification",   value=False)
    st.markdown('<div class="sidebar-section">Farmland Zone</div>', unsafe_allow_html=True)
    zone = st.text_input("Zone / Field ID", placeholder="e.g. Field-A3")
    cam  = st.text_input("Camera ID",       placeholder="e.g. CAM-07")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:.72rem;color:#3d5c42;line-height:1.6;">'
        'FarmGuard v2.4.1 · Model: YOLOv8<br>'
        'Last sync: just now · Status: <span style="color:#a3e635">●</span> Online'
        '</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🌾</div>
  <div>
    <div class="hero-title">FarmGuard</div>
    <div class="hero-sub">AI-Powered Animal Detection &amp; Farmland Protection</div>
  </div>
  <div class="live-badge">
    <div class="live-dot"></div>
    LIVE MONITORING
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "detection_log" not in st.session_state:
    animals_sample = list(ANIMALS.keys())
    now = datetime.now()
    log_rows = []
    for i in range(12):
        a = random.choice(animals_sample)
        t = now - timedelta(minutes=random.randint(1, 300))
        log_rows.append({
            "Time":       t.strftime("%H:%M:%S"),
            "Date":       t.strftime("%Y-%m-%d"),
            "Animal":     f"{ANIMALS[a]['emoji']}  {a}",
            "Threat":     ANIMALS[a]["threat"].capitalize(),
            "Confidence": f"{random.uniform(0.62, 0.98):.0%}",
            "Zone":       random.choice(["Field-A1","Field-A3","Field-B2","Field-C1"]),
            "Camera":     f"CAM-{random.randint(1,10):02d}",
        })
    st.session_state["detection_log"] = log_rows

if "total_scans" not in st.session_state:
    st.session_state["total_scans"] = random.randint(320, 580)
if "total_detections" not in st.session_state:
    st.session_state["total_detections"] = random.randint(40, 90)
if "alerts_sent" not in st.session_state:
    st.session_state["alerts_sent"] = random.randint(8, 25)

# ─────────────────────────────────────────────
#  METRIC CARDS
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Total Scans Today</div>
    <div class="metric-val">{st.session_state['total_scans']}</div>
    <div class="metric-delta">↑ 12% vs yesterday</div>
  </div>
  <div class="metric-card amber">
    <div class="metric-label">Animals Detected</div>
    <div class="metric-val">{st.session_state['total_detections']}</div>
    <div class="metric-delta">Across all zones</div>
  </div>
  <div class="metric-card red">
    <div class="metric-label">Threat Alerts Sent</div>
    <div class="metric-val">{st.session_state['alerts_sent']}</div>
    <div class="metric-delta">↑ 3 in last hour</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Model Accuracy</div>
    <div class="metric-val">94.7%</div>
    <div class="metric-delta">mAP@0.5 on farm dataset</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_detect, tab_live, tab_history, tab_analytics = st.tabs(
    ["🔍  Detect Image", "📡  Live Feed", "📋  Detection History", "📊  Analytics"])

# ── TAB 1 ──────────────────────────────────────
with tab_detect:
    col_upload, col_result = st.columns([1, 1], gap="large")
    with col_upload:
        st.markdown('<div class="section-title">Upload Image <div class="section-line"></div></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a farmland image or click to browse",
            type=["jpg","jpeg","png","bmp","webp"], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True,
                     caption=f"📸 {uploaded.name}  ·  {image.size[0]}×{image.size[1]}px")
            c1, c2 = st.columns(2)
            run_btn   = c1.button("⚡  Run Detection", use_container_width=True)
            clear_btn = c2.button("✕  Clear", use_container_width=True)
        else:
            st.markdown("""<div style="text-align:center;padding:3rem 1rem;color:#3d5c42;font-size:.9rem;">
                <div style="font-size:2.5rem;margin-bottom:.75rem;">📷</div>
                Upload a CCTV frame, drone image, or<br>camera trap photo to begin detection.</div>""",
                unsafe_allow_html=True)
            run_btn = False

    with col_result:
        st.markdown('<div class="section-title">Detection Results <div class="section-line"></div></div>', unsafe_allow_html=True)
        if uploaded and run_btn:
            with st.spinner("Running inference…"):
                time.sleep(1.6)
            n_det = random.randint(1, 4)
            detections = []
            used = set()
            for _ in range(n_det):
                while True:
                    a = random.choice(list(ANIMALS.keys()))
                    if a not in used: used.add(a); break
                conf = random.uniform(conf_threshold + 0.02, 0.98)
                w, h = image.size
                bx = random.randint(20, w//2)
                by = random.randint(20, h//2)
                bw = random.randint(60, w//3)
                bh = random.randint(60, h//3)
                detections.append({"animal": a, "conf": conf,
                    "box": (bx, by, bx+bw, by+bh), **ANIMALS[a]})

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            for d in detections:
                color = d["color"]
                draw.rectangle(d["box"], outline=color, width=3)
                label_text = f"{d['emoji']} {d['animal']}  {d['conf']:.0%}"
                bx1, by1 = d["box"][0], d["box"][1]
                draw.rectangle([bx1, by1-22, bx1+len(label_text)*8, by1], fill=color)
                try: draw.text((bx1+4, by1-18), label_text, fill=(0,0,0))
                except: pass
            st.image(annotated, use_container_width=True, caption="Annotated Output")

            for d in detections:
                badge_cls  = {"danger":"badge-danger","warn":"badge-warn","safe":"badge-safe"}[d["threat"]]
                threat_lbl = {"danger":"⚠ DANGEROUS","warn":"⚡ WARNING","safe":"✔ LOW RISK"}[d["threat"]]
                st.markdown(f"""
                <div class="detection-result">
                  <div class="det-header">
                    <span style="font-size:1.4rem">{d["emoji"]}</span>
                    <span class="det-name">{d["animal"]}</span>
                    <span class="det-badge {badge_cls}">{threat_lbl}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#6b8f70;margin-bottom:.3rem;">
                    <span>Confidence</span><span style="color:#e8f0e9;font-weight:600">{d['conf']:.1%}</span>
                  </div>
                  <div class="conf-bar-bg">
                    <div class="conf-bar" style="width:{d['conf']*100:.0f}%"></div>
                  </div>
                  <div style="font-size:.75rem;color:#3d5c42;margin-top:.5rem;">
                    BBox: ({d['box'][0]}, {d['box'][1]}) → ({d['box'][2]}, {d['box'][3]})
                  </div>
                </div>""", unsafe_allow_html=True)

            for d in [x for x in detections if x["threat"]=="danger"] if alert_danger else []:
                st.markdown(f"""<div class="alert-pill">
                  <span style="font-size:1.1rem">🚨</span>
                  <span><strong>THREAT ALERT:</strong> {d['emoji']} {d['animal']} detected in {zone or 'Unknown Zone'}
                  — Confidence {d['conf']:.0%}</span></div>""", unsafe_allow_html=True)
                st.session_state["alerts_sent"] += 1

            st.session_state["total_scans"] += 1
            st.session_state["total_detections"] += n_det
            now = datetime.now()
            for d in detections:
                st.session_state["detection_log"].insert(0, {
                    "Time": now.strftime("%H:%M:%S"), "Date": now.strftime("%Y-%m-%d"),
                    "Animal": f"{d['emoji']}  {d['animal']}",
                    "Threat": d["threat"].capitalize(),
                    "Confidence": f"{d['conf']:.0%}",
                    "Zone": zone or "Unknown", "Camera": cam or "Unknown",
                })
            st.success(f"✅  Detected **{n_det}** animal(s) · Inference time: {random.uniform(0.08,0.35):.3f}s")
        elif not uploaded:
            st.markdown("""<div style="text-align:center;padding:3rem 1rem;color:#3d5c42;font-size:.9rem;">
                <div style="font-size:2.5rem;margin-bottom:.75rem;">🔍</div>
                Detection results will appear here<br>after you upload and run an image.</div>""",
                unsafe_allow_html=True)

# ── TAB 2 ──────────────────────────────────────
with tab_live:
    st.markdown('<div class="section-title">Live Camera Feeds <div class="section-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
        <div class="live-dot"></div>
        <span style="font-size:.8rem;font-weight:600;color:#a3e635;letter-spacing:.08em;">STREAMING ACTIVE</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
        <div style="background:var(--bg-raised);border-radius:10px;padding:1rem;text-align:center;">
          <div style="font-size:.7rem;color:#3d5c42;letter-spacing:.1em;margin-bottom:.5rem;">CAM-01 · FIELD-A1</div>
          <div style="font-size:3rem;">🌿</div>
          <div style="font-size:.75rem;color:#6b8f70;margin-top:.5rem;">No animal detected</div>
          <div style="font-size:.65rem;color:#3d5c42;">FPS: 24 · 1080p</div>
        </div>
        <div style="background:var(--bg-raised);border-radius:10px;padding:1rem;text-align:center;border:1px solid rgba(239,68,68,.3);">
          <div style="font-size:.7rem;color:#ef4444;letter-spacing:.1em;margin-bottom:.5rem;">⚠ CAM-03 · FIELD-A3</div>
          <div style="font-size:3rem;">🐗</div>
          <div style="font-size:.75rem;color:#f87171;margin-top:.5rem;font-weight:600;">Wild Boar — 91%</div>
          <div style="font-size:.65rem;color:#3d5c42;">FPS: 24 · 1080p</div>
        </div>
        <div style="background:var(--bg-raised);border-radius:10px;padding:1rem;text-align:center;">
          <div style="font-size:.7rem;color:#3d5c42;letter-spacing:.1em;margin-bottom:.5rem;">CAM-07 · FIELD-C1</div>
          <div style="font-size:3rem;">🌿</div>
          <div style="font-size:.75rem;color:#6b8f70;margin-top:.5rem;">No animal detected</div>
          <div style="font-size:.65rem;color:#3d5c42;">FPS: 18 · 720p</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄  Refresh Feeds", use_container_width=True): st.toast("Feeds refreshed!", icon="✅")
    with c2:
        if st.button("📷  Capture Snapshot", use_container_width=True): st.toast("Snapshot saved!", icon="📸")

# ── TAB 3 ──────────────────────────────────────
with tab_history:
    st.markdown('<div class="section-title">Detection Log <div class="section-line"></div></div>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3, _ = st.columns([1,1,1,2])
    with col_f1: filter_threat = st.selectbox("Threat Level", ["All","Danger","Warn","Safe"])
    with col_f2: filter_zone   = st.selectbox("Zone", ["All","Field-A1","Field-A3","Field-B2","Field-C1"])
    with col_f3:
        if st.button("🗑  Clear Log"): st.session_state["detection_log"] = []; st.rerun()
    log = st.session_state["detection_log"]
    if filter_threat != "All": log = [r for r in log if r["Threat"].lower() == filter_threat.lower()]
    if filter_zone   != "All": log = [r for r in log if r["Zone"] == filter_zone]
    if log:
        df = pd.DataFrame(log)
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)
        st.markdown(f'<div style="font-size:.75rem;color:#3d5c42;margin-top:.5rem;">Showing {len(log)} records</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#3d5c42;text-align:center;padding:2rem;">No records match your filters.</div>', unsafe_allow_html=True)

# ── TAB 4 ──────────────────────────────────────
with tab_analytics:
    st.markdown('<div class="section-title">Detection Analytics <div class="section-line"></div></div>', unsafe_allow_html=True)
    log_all = st.session_state["detection_log"]
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown("**Detections by Animal**")
        if log_all:
            animal_names = [r["Animal"].split("  ")[-1].strip() for r in log_all]
            counts = pd.Series(animal_names).value_counts().reset_index()
            counts.columns = ["Animal","Count"]
            st.bar_chart(counts.set_index("Animal"), color="#a3e635", use_container_width=True)
        else: st.info("No data yet.")
    with col_ch2:
        st.markdown("**Detections by Threat Level**")
        if log_all:
            threats = [r["Threat"] for r in log_all]
            tc = pd.Series(threats).value_counts().reset_index()
            tc.columns = ["Threat","Count"]
            st.bar_chart(tc.set_index("Threat"), color="#f59e0b", use_container_width=True)
        else: st.info("No data yet.")
    st.markdown("**Hourly Detection Activity (simulated)**")
    hours = list(range(0,24))
    activity_df = pd.DataFrame({"Hour": hours,
        "Detections": [random.randint(0,12) for _ in hours]}).set_index("Hour")
    st.bar_chart(activity_df, color="#a3e635", use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""<div class="footer">
    FarmGuard · AI Animal Detection Platform · v2.4.1<br>
    Powered by YOLOv8 &amp; Streamlit · © 2025 FarmGuard Labs
</div>""", unsafe_allow_html=True)