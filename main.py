from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)



model = YOLO("../models/n_version_1_3.pt")



classNames = ["fake", "real"]
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()


    results = model(img, stream=True, verbose=False)
    MIN_CONFIDENCE = 0.5
    MIN_BOX_AREA = 1000
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            area = w * h
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if area > MIN_BOX_AREA and conf > MIN_CONFIDENCE and 0 <= cls < len(classNames):
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(0, y1)), scale=1, thickness=2)
            else:
                cvzone.putTextRect(img, 'Too Far or Uncertain', (x1, y1), scale=1, thickness=2)


    fps = 1 / (new_frame_time- prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()