


"""
#roi 사다리꼴 영역

import cv2
import numpy as np
from ultralytics import YOLO

# 1. YOLOv8 모델 불러오기
model = YOLO("/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt")


# 2. 비디오 열기
cap = cv2.VideoCapture(r"/home/hkit/Downloads/Driving Downtown Seoul.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # 3. 도로 ROI 사다리꼴 좌표 정의 (해상도 맞게 조정)
    roi_points = np.array([
        (int(width * 0.43), int(height * 0.65)),  # 좌상
        (int(width * 0.58), int(height * 0.65)),  # 우상
        (int(width * 0.85), int(height * 0.95)),  # 우하
        (int(width * 0.23), int(height * 0.95))   # 좌하
    ])

    # 4. ROI 반투명 영역 표시
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_points], color=(255, 0, 0))  # 파란색 영역
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # 5. YOLO 객체 탐지
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        # 6. 바운딩 박스 좌표 기반 객체가 ROI 안에 있는지 판단
        bbox_polygon = np.array([
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ])

        # 7. 바운딩 박스와 ROI가 겹치는지 여부
        mask_roi = np.zeros((height, width), dtype=np.uint8)
        mask_bbox = np.zeros((height, width), dtype=np.uint8)

        cv2.fillPoly(mask_roi, [roi_points], 255)
        cv2.fillPoly(mask_bbox, [bbox_polygon], 255)

        # AND 마스킹 → 겹치는 영역이 있으면 in_roi = True
        intersection = cv2.bitwise_and(mask_roi, mask_bbox)
        in_roi = cv2.countNonZero(intersection) > 0

        # 8. ROI 안이면 빨간색, 아니면 초록색
        color = (0, 0, 255) if in_roi else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # 9. 화면 출력
    cv2.imshow("YOLOv8 with Road ROI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""


import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt")
cap = cv2.VideoCapture(r"/home/hkit/Downloads/Driving Downtown Seoul.mp4")

def interpolate(p1, p2, ratio):
    return (
        int(p1[0] + (p2[0] - p1[0]) * ratio),
        int(p1[1] + (p2[1] - p1[1]) * ratio)
    )


zones = [0, 1/2, 1.0]
colors = [(0, 255, 255), (0, 0, 255)]  # 좌-중-우
ellipse_color = (100, 255, 100)  # 연한 초록색

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    warning = False  # ← 프레임마다 초기화

    # 꼭짓점
    pt1 = (int(width * 0.35), int(height * 0.8))   # 좌상
    pt2 = (int(width * 0.7), int(height * 0.8))   # 우상
    pt3 = (int(width * 0.9), int(height * 0.99))   # 우하
    pt4 = (int(width * 0.2), int(height * 0.99))   # 좌하

    # 마스크 초기화
    mask_roi = np.zeros((height, width), dtype=np.uint8)
    trapezoid_overlay = np.zeros_like(frame)
    ellipse_overlay = np.zeros_like(frame)

    # 사다리꼴을 가로 방향으로 3구간 나누기
    for i in range(2):
    # 상단 라인: pt1 → pt2 (좌상 → 우상)
        top_l = interpolate(pt1, pt4, zones[i])
        top_r = interpolate(pt1, pt4, zones[i + 1])

        # 하단 라인: pt4 → pt3 (좌하 → 우하)
        bot_l = interpolate(pt2, pt3, zones[i])
        bot_r = interpolate(pt2, pt3, zones[i + 1])
    
        zone_pts = np.array([top_l, top_r, bot_r, bot_l], dtype=np.int32)
        cv2.fillPoly(trapezoid_overlay, [zone_pts], colors[i])
        cv2.fillPoly(mask_roi, [zone_pts], 255)


    # 반원(타원형 ROI) 따로 그리기
    center = (width // 2, height)
    axes = (int(width * 0.5), int(height * 0.3))
    cv2.ellipse(ellipse_overlay, center, axes, 0, 180, 360, ellipse_color, -1)
    cv2.ellipse(mask_roi, center, axes, 0, 180, 360, 255, -1)

    # 시각화용 오버레이 합성
    combined_overlay = cv2.addWeighted(trapezoid_overlay, 0.5, ellipse_overlay, 0.5, 0)
    frame = cv2.addWeighted(combined_overlay, 0.4, frame, 1.0, 0)

    # YOLO 객체 탐지
    results = model(frame)[0]
    

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        # 객체 박스 마스크
        mask_bbox = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask_bbox, (x1, y1), (x2, y2), 100, -1)

        # ROI와 겹치는지 확인
        intersection = cv2.bitwise_and(mask_roi, mask_bbox)
        in_roi = cv2.countNonZero(intersection) > 0

        if in_roi:
            warning = True #하나라도 겹치면 경고 표시

        color = (0, 255, 0) if in_roi else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if warning:
            cv2.putText(frame, "Safe", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # 결과 출력
    cv2.imshow("YOLOv8 with Horizontal Zones + Ellipse ROI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
import cv2
import numpy as np
from ultralytics import YOLO

# 1. YOLO 모델 불러오기
model = YOLO("/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt")

# 2. 비디오 열기
video_path = r"/home/hkit/Downloads/Driving Downtown Seoul.mp4"
cap = cv2.VideoCapture(video_path)

# 3. FPS 정보
fps = cap.get(cv2.CAP_PROP_FPS)
original_delay = int(1000 / fps)
delay = int(original_delay * 1)

# ✅ 4. 차선 인식 함수
def detect_lanes(frame):
    height, width = frame.shape[:2]

    # 1) 그레이 변환, 블러, Canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2) ROI 마스크 (하단 사다리꼴)
    mask = np.zeros_like(edges)
    roi = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 3) 허프 변환으로 선 추출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

    # 4) 선 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 노란색

    return frame

# ✅ 5. 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model.predict(source=frame, conf=0.3, verbose=False)
    r = results[0]

    # YOLO 결과 시각화
    annotated_frame = r.plot()

    # OpenCV 차선 인식 적용
    lane_frame = detect_lanes(annotated_frame)

    # 프레임 리사이즈 (선택)
    resized = cv2.resize(lane_frame, (800, 600))

    # 출력
    cv2.imshow("YOLO + Lane Detection", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
"""
