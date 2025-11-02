import streamlit as st
import cv2
import tempfile
import os
import math
import numpy as np
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
import imghdr

import os
sender_email = os.getenv("EMAIL_USER")
sender_password = os.getenv("EMAIL_PASS")

# --- Load YOLO models once ---
@st.cache_resource
def load_models():
    model_general = YOLO("cheating.pt")
    model_handsign = YOLO("best_handsign_detector.pt")
    model_chit = YOLO("best_chit_detection.pt")
    model_phone = YOLO("phone.pt")
    return model_general, model_handsign, model_chit, model_phone

model_general, model_handsign, model_chit, model_phone = load_models()

# --- Tracking state ---
next_person_id = 0
tracked_objects = {}  # id -> (x_center, y_center)
cheating_log = {}    # id -> {"count": int, "snapshots": [np.array]}

MAX_IDS = 7  # maximum number of unique person IDs

def assign_id(x1, y1, x2, y2, tracked_objects, threshold=50):
    global next_person_id
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Try to match with existing IDs
    min_dist = float("inf")
    assigned_id = None
    for obj_id, (px, py) in tracked_objects.items():
        dist = math.hypot(cx - px, cy - py)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            assigned_id = obj_id

    # If no match and we still have capacity, create new ID
    if assigned_id is None:
        if next_person_id < MAX_IDS:
            assigned_id = next_person_id
            next_person_id += 1
            tracked_objects[assigned_id] = (cx, cy)
        else:
            # Already reached max IDs → ignore this detection
            return None

    # Update tracked object position if valid
    if assigned_id is not None:
        tracked_objects[assigned_id] = (cx, cy)

    return assigned_id

def log_cheating(person_id, frame, bbox):
    x1, y1, x2, y2 = bbox
    face_crop = frame[y1:y2, x1:x2].copy()
    if person_id not in cheating_log:
        cheating_log[person_id] = {"count": 1, "snapshot": face_crop}
    else:
        cheating_log[person_id]["count"] += 1
        # do NOT overwrite snapshot, keep the first one

def run_inference(frame):
    annotated = frame.copy()

    # General detections (green) with IDs
    results_general = model_general(frame, conf=0.25)
    for box in results_general[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        person_id = assign_id(x1, y1, x2, y2, tracked_objects)
        label = f"ID {person_id} | {model_general.names[cls_id]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if model_general.names[cls_id].lower() in ["cheating", "suspicious"]:
            log_cheating(person_id, frame, (x1, y1, x2, y2))

    # Handsign detections (red)
    results_handsign = model_handsign(frame, conf=0.40)
    for box in results_handsign[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_id = assign_id(x1, y1, x2, y2, tracked_objects)
        label = f"{model_handsign.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        log_cheating(person_id, frame, (x1, y1, x2, y2))

    # Chit detections (blue)
    results_chit = model_chit(frame, conf=0.40)
    for box in results_chit[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_id = assign_id(x1, y1, x2, y2, tracked_objects)
        label = f"{model_chit.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        log_cheating(person_id, frame, (x1, y1, x2, y2))

    # Phone detections (purple)
    results_phone = model_phone(frame, conf=0.40)
    for box in results_phone[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_id = assign_id(x1, y1, x2, y2, tracked_objects)
        label = f"{model_phone.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        log_cheating(person_id, frame, (x1, y1, x2, y2))

    return annotated

def send_report_via_email(sender_email, sender_password, recipient_email):
    """Send cheating report via SMTP with snapshots attached."""
    msg = EmailMessage()
    msg["Subject"] = "Cheating Detection Report"
    msg["From"] = sender_email
    msg["To"] = 'prasanna.22310087@viit.ac.in'

    # Build text summary
    body = "Cheating Report Summary:\n\n"
    for pid, data in cheating_log.items():
        body += f"Student ID {pid}: {data['count']} times\n"
    msg.set_content(body)

    # Attach snapshots
    for pid, data in cheating_log.items():
        for i, snap in enumerate(data["snapshots"]):
            _, img_bytes = cv2.imencode(".jpg", snap)
            img_data = img_bytes.tobytes()
            msg.add_attachment(img_data, maintype="image", subtype=imghdr.what(None, img_data),
                               filename=f"student{pid}_snap{i+1}.jpg")

    # Send via Gmail SMTP (example)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

def send_report_via_email(sender_email, sender_password, recipient_email):
    """Send cheating report via SMTP with one snapshot per student attached."""
    msg = EmailMessage()
    msg["Subject"] = "Cheating Detection Report"
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Build text summary
    body = "Cheating Report Summary:\n\n"
    for pid, data in cheating_log.items():
        body += f"Student ID {pid}: {data['count']} times\n"
    msg.set_content(body)

    # Attach one snapshot per student
    for pid, data in cheating_log.items():
        if "snapshot" in data:
            snap = data["snapshot"]
            _, img_bytes = cv2.imencode(".jpg", snap)
            img_data = img_bytes.tobytes()
            msg.add_attachment(
                img_data,
                maintype="image",
                subtype=imghdr.what(None, img_data),
                filename=f"student{pid}_snapshot.jpg"
            )

    # Send via Gmail SMTP
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

# --- Streamlit UI ---
st.title("📹 Multi-Model Cheating Detection with Tracking & Report")

uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        annotated = run_inference(frame)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Processed Image")

    elif "video" in file_type:
        if "video_processed" not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            cap = cv2.VideoCapture(temp_path)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = run_inference(frame)
                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            cap.release()
            try:
                os.remove(temp_path)
            except PermissionError:
                pass

        st.session_state["video_processed"] = True

        
        # --- After video ends, show cheating report ---
        st.subheader("📊 Cheating Report")
    if cheating_log:
        for pid, data in cheating_log.items():
            st.markdown(f"**Student ID {pid}** — caught {data['count']} times")
            snap_rgb = cv2.cvtColor(data["snapshot"], cv2.COLOR_BGR2RGB)
            st.image(snap_rgb, caption=f"Student {pid} Snapshot", use_column_width=True)

    # Email send button (no password entry in UI)
        if st.button("Send Report via Email"):
            try:
                send_report_via_email(sender_email, sender_password, "prasanna.22310087@viit.ac.in")
                st.success("Report sent successfully")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
    else:
        st.info("No cheating detected in this video.")

            # --- Email form ---
    st.subheader("📧 Send Report via Email")
    with st.form("email_form"):
        recipient = st.text_input("Recipient Email")
        sender = st.text_input("Your Email (Gmail)", type="default")
        password = st.text_input("App Password", type="password")
        submitted = st.form_submit_button("Send Report")

        if submitted:
            try:
                send_report_via_email(sender, password, recipient)
                st.success(f"Report sent successfully to {recipient}")
            except Exception as e:
                st.error(f"Failed to send email: {e}")


