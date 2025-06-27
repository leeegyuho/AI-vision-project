
# SAFE, CAUTION, DANGER 경고 알림
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt")
cap = cv2.VideoCapture(r"/home/hkit/Downloads/Driving Downtown Seoul.mp4")

def interpolate(p1, p2, ratio):
    return (
        int(p1[0] + (p2[0] - p1[0]) * ratio),
        int(p1[1] + (p2[1] - p1[1]) * ratio)
    )

zones = [0, 1/2, 1.0]

# 색상 정의
yellow = (0, 255, 255)
red = (0, 0, 255)
green = (100, 255, 100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # 꼭짓점 설정
    pt1 = (int(width * 0.35), int(height * 0.8))
    pt2 = (int(width * 0.7), int(height * 0.8))
    pt3 = (int(width * 0.9), int(height * 0.99))
    pt4 = (int(width * 0.2), int(height * 0.99))

    mask_red = np.zeros((height, width), dtype=np.uint8)
    mask_yellow = np.zeros((height, width), dtype=np.uint8)
    mask_safe = np.zeros((height, width), dtype=np.uint8)

    trapezoid_overlay = np.zeros_like(frame)
    ellipse_overlay = np.zeros_like(frame)

    # 두 사다리꼴 (노랑, 빨강)
    for i in range(2):
        top_l = interpolate(pt1, pt4, zones[i])
        top_r = interpolate(pt1, pt4, zones[i + 1])
        bot_l = interpolate(pt2, pt3, zones[i])
        bot_r = interpolate(pt2, pt3, zones[i + 1])

        zone_pts = np.array([top_l, top_r, bot_r, bot_l], dtype=np.int32)

        color = yellow if i == 0 else red
        cv2.fillPoly(trapezoid_overlay, [zone_pts], color)

        if i == 0:
            cv2.fillPoly(mask_yellow, [zone_pts], 255)
        else:
            cv2.fillPoly(mask_red, [zone_pts], 255)

    # 반원 (안전 구역)
    center = (width // 2, height)
    axes = (int(width * 0.5), int(height * 0.3))
    cv2.ellipse(ellipse_overlay, center, axes, 0, 180, 360, green, -1)
    cv2.ellipse(mask_safe, center, axes, 0, 180, 360, 255, -1)

    # 오버레이 합성
    combined_overlay = cv2.addWeighted(trapezoid_overlay, 0.5, ellipse_overlay, 0.5, 0)
    frame = cv2.addWeighted(combined_overlay, 0.4, frame, 1.0, 0)

    # 객체 탐지
    results = model(frame)[0]

    status = None  # 'danger', 'warning', 'safe'

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        mask_bbox = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask_bbox, (x1, y1), (x2, y2), 100, -1)

        if cv2.countNonZero(cv2.bitwise_and(mask_red, mask_bbox)) > 0:
            status = 'danger'
            break
        elif cv2.countNonZero(cv2.bitwise_and(mask_yellow, mask_bbox)) > 0:
            status = 'warning'
        elif cv2.countNonZero(cv2.bitwise_and(mask_safe, mask_bbox)) > 0 and status is None:
            status = 'safe'

        color = (0, 0, 255) if status == 'danger' else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 상태 메시지
    if status == 'danger':
        cv2.putText(frame, "!!DANGER!!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    elif status == 'warning':
        cv2.putText(frame, "CAUTION", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
    elif status == 'safe':
        cv2.putText(frame, "SAFE", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 3)

    # 결과 표시
    cv2.imshow("YOLOv8 - Collision Warning System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
