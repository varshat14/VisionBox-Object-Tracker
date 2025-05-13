import cv2
from ultralytics import YOLO

print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

def generate_frames():
    print("[INFO] Starting video capture...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Camera not detected!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Frame read failed!")
            break

        results = model(frame, verbose=False)[0]
        print(f"[INFO] {len(results.boxes)} objects detected")

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
