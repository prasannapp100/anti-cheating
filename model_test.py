import cv2
from ultralytics import YOLO
import math

# Load YOLO models
model_general = YOLO("cheating.pt")
model_handsign = YOLO("handsign_detection//best_handsign_detector.pt")

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Tracking state
next_person_id = 0
tracked_objects = {}  # id -> (x_center, y_center)

def assign_id(x1, y1, x2, y2, tracked_objects, threshold=50):
    """
    Assigns an ID to a bounding box based on proximity to previous boxes.
    threshold = max distance (pixels) to consider same object.
    """
    global next_person_id
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Find closest existing object
    min_dist = float("inf")
    assigned_id = None
    for obj_id, (px, py) in tracked_objects.items():
        dist = math.hypot(cx - px, cy - py)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            assigned_id = obj_id

    # If no match, create new ID
    if assigned_id is None:
        assigned_id = next_person_id
        next_person_id += 1

    # Update tracked object position
    tracked_objects[assigned_id] = (cx, cy)
    return assigned_id

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame (end of video or wrong path).")
        break

    # Run YOLO inference
    results_general = model_general(frame, conf=0.25)
    results_phone   = model_handsign(frame, conf=0.40)

    annotated_frame = frame.copy()

    # Draw general detections (green) with IDs
    for box in results_general[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        person_id = assign_id(x1, y1, x2, y2, tracked_objects)

        label = f"ID {person_id} | {model_general.names[cls_id]} {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw phone/handsign detections (red)
    for box in results_phone[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model_handsign.names[cls_id]} {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("YOLO Multi-Model Demo with IDs", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
