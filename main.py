import cv2
import numpy as np
from collections import deque
import time

# מקסימום נקודות לשמירה בשובל
MAX_TRAIL = 50
trail_points = deque(maxlen=MAX_TRAIL)

# טווח כחול
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# פונקציה למעבר צבע ל-HSV וליצירת מסכה
def get_mask(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)

# פונקציה למציאת הקונטור הגדול ביותר
def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest if cv2.contourArea(largest) > 300 else None

# פונקציה לציור השובל עם דעיכה וצבע משתנה
def draw_trail(frame, points):
    for i in range(1, len(points)):
        color = (
            int(255 * (i / len(points))),       # B
            int(128 * (1 - i / len(points))),   # G
            int(255 * (1 - i / len(points)))    # R
        )
        thickness = int(5 * (1 - i / len(points))) + 1
        cv2.line(frame, points[i - 1], points[i], color, thickness)

# פתיחת מצלמה
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = get_mask(frame, lower_blue, upper_blue)
    largest = get_largest_contour(mask)

    if largest is not None:
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # כחול
        center = (x + w // 2, y + h // 2)
        trail_points.appendleft(center)

    draw_trail(frame, trail_points)

    # חישוב FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Press 'Q' to Quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Blue Object Tracking - Pro", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

