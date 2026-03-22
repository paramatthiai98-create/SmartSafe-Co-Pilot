import streamlit as st
import random
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import cv2
import tempfile
import time
from PIL import Image, ImageDraw
import numpy as np
import os

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="SmartSafe Co-Pilot Dashboard",
    layout="wide"
)

# auto refresh
st_autorefresh(interval=2000, key="datarefresh")

# ------------------------
# CUSTOM CSS
# ------------------------
st.markdown("""
<style>
.main {
    background-color: #050b18;
    color: white;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: white !important;
}
.card {
    background: #0d1528;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    min-height: 180px;
}
.small-card {
    background: #101a31;
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 10px;
}
.risk-safe {
    background-color: rgba(20, 120, 60, 0.45);
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}
.risk-warning {
    background-color: rgba(180, 140, 0, 0.45);
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}
.risk-danger {
    background-color: rgba(180, 30, 30, 0.45);
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
}
.metric-title {
    font-size: 14px;
    opacity: 0.8;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# LOAD MODEL
# ------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------
# DEFAULT ICON GENERATOR
# ------------------------
def create_default_icon(text, bg_color, size=(90, 90)):
    img = Image.new("RGBA", size, bg_color)
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, size[0]-8, size[1]-8), fill=(255, 255, 255, 40))
    draw.text((size[0]//3, size[1]//3), text, fill="white")
    return img

@st.cache_resource
def load_icons():
    """
    ถ้ามีไฟล์ helmet.png / no_helmet.png ในโฟลเดอร์เดียวกับ app.py จะใช้ไฟล์จริง
    ถ้าไม่มี จะสร้าง icon placeholder ให้เอง
    """
    if os.path.exists("helmet.jpg"):
        helmet_icon = Image.open("helmet.jpg").convert("RGBA")
    else:
        helmet_icon = create_default_icon("H", (0, 170, 80, 255))

    if os.path.exists("no_helmet.jpg"):
        no_helmet_icon = Image.open("no_helmet.jpg").convert("RGBA")
    else:
        no_helmet_icon = create_default_icon("X", (220, 40, 40, 255))

    return helmet_icon, no_helmet_icon

helmet_icon, no_helmet_icon = load_icons()

# ------------------------
# MACHINE DATA
# ------------------------
def generate_data():
    return {
        "vibration": random.randint(0, 100),
        "temperature": random.randint(25, 80)
    }

# ------------------------
# RISK CALCULATION
# ------------------------
def calculate_risk(helmet, distance, vibration):
    risk = 0
    reasons = []

    if not helmet:
        risk += 30
        reasons.append("No helmet detected")

    if distance < 30:
        risk += 40
        reasons.append("Worker too close to machine")

    if vibration > 70:
        risk += 35
        reasons.append("High machine vibration")

    return risk, reasons

def decision_logic(risk):
    if risk > 80:
        return "HIGH RISK", "STOP MACHINE"
    elif risk > 50:
        return "WARNING", "CHECK SYSTEM"
    else:
        return "SAFE", "NORMAL OPERATION"

# ------------------------
# OVERLAY ICON
# ------------------------
def overlay_icon(frame, icon, x, y, size=55):
    icon = icon.resize((size, size))
    icon_np = np.array(icon)

    h, w = icon_np.shape[:2]

    if x >= frame.shape[1] or y >= frame.shape[0]:
        return frame

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    icon_crop = icon_np[0:y2-y1, 0:x2-x1]

    if icon_crop.shape[2] < 4:
        return frame

    alpha = icon_crop[:, :, 3] / 255.0

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha * icon_crop[:, :, c] +
            (1 - alpha) * frame[y1:y2, x1:x2, c]
        ).astype(np.uint8)

    return frame

# ------------------------
# TITLE
# ------------------------
st.markdown('<div class="section-title">SmartSafe Co-Pilot Dashboard</div>', unsafe_allow_html=True)

# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox(
    "Demo Mode",
    ["Auto", "Force Safe", "Force Risk"]
)

icon_size = st.sidebar.slider("Icon Size", 30, 100, 55)
frame_delay = st.sidebar.slider("Playback Speed Delay (sec)", 0.01, 0.20, 0.05)

uploaded_file = st.file_uploader("📹 Upload Video", type=["mp4", "mov", "avi"])

# optional custom icons
st.sidebar.markdown("### Optional Custom Icons")
helmet_upload = st.sidebar.file_uploader("Upload Helmet Icon", type=["png"], key="helmet_icon")
no_helmet_upload = st.sidebar.file_uploader("Upload No-Helmet Icon", type=["png"], key="nohelmet_icon")

if helmet_upload is not None:
    helmet_icon = Image.open(helmet_upload).convert("RGBA")

if no_helmet_upload is not None:
    no_helmet_icon = Image.open(no_helmet_upload).convert("RGBA")

# ------------------------
# SESSION STATE
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "latest_people" not in st.session_state:
    st.session_state.latest_people = 0

if "latest_helmet" not in st.session_state:
    st.session_state.latest_helmet = "NO"

if "latest_distance" not in st.session_state:
    st.session_state.latest_distance = 0

if "latest_vibration" not in st.session_state:
    st.session_state.latest_vibration = 0

if "latest_temperature" not in st.session_state:
    st.session_state.latest_temperature = 0

if "latest_risk" not in st.session_state:
    st.session_state.latest_risk = 0

if "latest_status" not in st.session_state:
    st.session_state.latest_status = "SAFE"

if "latest_action" not in st.session_state:
    st.session_state.latest_action = "NORMAL OPERATION"

if "latest_reasons" not in st.session_state:
    st.session_state.latest_reasons = []

# ------------------------
# PLACEHOLDERS
# ------------------------
top1, top2, top3 = st.columns(3)
video_placeholder = st.empty()
chart_placeholder = st.empty()

# ------------------------
# DASHBOARD HEADER CARDS
# ------------------------
def render_dashboard():
    with top1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Worker Status")
        st.write(f"People detected: {st.session_state.latest_people}")
        st.write(f"Helmet: {st.session_state.latest_helmet}")
        st.write(f"Distance: {st.session_state.latest_distance} cm")

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(helmet_icon, width=70, caption="Helmet")
        with col_b:
            st.image(no_helmet_icon, width=70, caption="No Helmet")
        st.markdown('</div>', unsafe_allow_html=True)

    with top2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Machine Status")
        st.write(f"Vibration: {st.session_state.latest_vibration}")
        st.write(f"Temperature: {st.session_state.latest_temperature} °C")
        st.markdown('</div>', unsafe_allow_html=True)

    with top3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk Analysis")
        st.markdown(
            f'<div class="metric-title">Risk Score</div><div class="metric-value">{st.session_state.latest_risk}</div>',
            unsafe_allow_html=True
        )

        if st.session_state.latest_status == "HIGH RISK":
            st.markdown(f'<div class="risk-danger">{st.session_state.latest_status}</div>', unsafe_allow_html=True)
        elif st.session_state.latest_status == "WARNING":
            st.markdown(f'<div class="risk-warning">{st.session_state.latest_status}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-safe">{st.session_state.latest_status}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("AI Decision Support")
    st.write(f"Recommended Action: **{st.session_state.latest_action}**")

    st.subheader("Explainable AI")
    if st.session_state.latest_reasons:
        for reason in sorted(set(st.session_state.latest_reasons)):
            st.write(f"- {reason}")
    else:
        st.write("- No major risk detected")

def render_chart():
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        chart_placeholder.line_chart(df, y="risk")
    else:
        chart_placeholder.empty()

# initial draw
render_dashboard()
render_chart()

# ------------------------
# PROCESS VIDEO
# ------------------------
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO DETECTION
        results = model(frame)[0]
        annotated = frame.copy()

        # get persons only
        person_boxes = []
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            if label == "person":
                person_boxes.append(box)

        total_risk = 0
        reasons_all = []
        helmet_count = 0
        no_helmet_count = 0
        latest_distance = 0

        d = generate_data()

        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # helmet logic
            if mode == "Force Safe":
                helmet = True
            elif mode == "Force Risk":
                helmet = False
            else:
                helmet = (i % 2 == 0)

            # fake distance
            distance = random.randint(10, 100)
            latest_distance = distance

            risk, reasons = calculate_risk(helmet, distance, d["vibration"])
            total_risk += risk
            reasons_all.extend(reasons)

            if helmet:
                helmet_count += 1
                color = (0, 255, 0)
                text = "Helmet"
            else:
                no_helmet_count += 1
                color = (0, 0, 255)
                text = "No Helmet"

            # bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{text} | {distance} cm",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # icon overlay near head
            head_x = x1
            head_y = max(y1 - 60, 0)

            if helmet:
                annotated = overlay_icon(annotated, helmet_icon, head_x, head_y, size=icon_size)
            else:
                annotated = overlay_icon(annotated, no_helmet_icon, head_x, head_y, size=icon_size)

        # avg risk
        if len(person_boxes) > 0:
            avg_risk = int(total_risk / len(person_boxes))
        else:
            avg_risk = 0

        status, action = decision_logic(avg_risk)

        # latest helmet summary
        if no_helmet_count > 0:
            helmet_text = "NO"
        elif helmet_count > 0:
            helmet_text = "YES"
        else:
            helmet_text = "NO PERSON"

        # save session state
        st.session_state.latest_people = len(person_boxes)
        st.session_state.latest_helmet = helmet_text
        st.session_state.latest_distance = latest_distance
        st.session_state.latest_vibration = d["vibration"]
        st.session_state.latest_temperature = d["temperature"]
        st.session_state.latest_risk = avg_risk
        st.session_state.latest_status = status
        st.session_state.latest_action = action
        st.session_state.latest_reasons = reasons_all

        st.session_state.history.append({"risk": avg_risk})
        if len(st.session_state.history) > 30:
            st.session_state.history = st.session_state.history[-30:]

        # render dashboard
        render_dashboard()
        render_chart()

        # show video
        video_placeholder.image(annotated, channels="BGR", use_container_width=True)

        time.sleep(frame_delay)

    cap.release()

else:
    st.info("กรุณาอัปโหลดวิดีโอเพื่อเริ่มตรวจจับคนใส่/ไม่ใส่หมวกนิรภัย")
