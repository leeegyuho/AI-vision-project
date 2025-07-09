import cv2
import numpy as np
from ultralytics import YOLO
import time


# --------------------------- 초기화 및 트랙바 설정 ---------------------------

# YOLOv8 모델 로드 및 초기화
model = YOLO('/home/hkit/Pictures/test/yolov8_custom14/weights/best.pt')
model.eval() # 모델을 추론 모드로 전환 : 정확도 향상
model.fuse() # eval() 다음 사용, 최적화
_ = model(np.zeros((360, 640, 3), dtype=np.uint8))  # 첫 프레임에서 딜레이 발생할 경우를 위해
                                                    # 모델 warm-up (빈 프레임으로 1회 호출)
                                                    # model.fuse() 사용시 필요

# 동영상 파일 로드 및 파라미터 설정
# .get()으로 프레임 수, FPS 등 정보 얻을 수 있음
# cap = cv2.VideoCapture('/home/hkit/Pictures/video/rural_cut.webm')
cap = cv2.VideoCapture('/home/hkit/Pictures/video/output_video.mp4')
resize_width, resize_height = 640, 360
fps = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 전체 프레임 수  
                                                      
delay = int(1000 // (fps * 15))  # OpenCV에서 cv2.waitKey()로 딜레이 조절, 영상 속도 n배속

# 사용자 정의 클래스 이름 매핑
class_names = {
    0: "vehicle", 1: "big vehicle", 4: "bike",
    5: "human", 6: "animals", 7: "obstacles"
}

# 카메라 및 ROI 관련 파라미터
focal_length = 630
CAMERA_TO_BUMPER_OFFSET = 1.0  # 실제 거리 계산 시 차량 전면까지 거리 보정

# ROI의 위쪽/아래쪽 y좌표 및 마스크 포함 임계값 설정
danger_bottom, danger_top = 360, 300
warning_bottom, warning_top = 360, 260
danger_threshold = 0.1 
warning_threshold = 0.1

# ROI 및 FPS 관련 전역 변수 초기화
prev_frame_time = 0
prev_edges = None
frame_count = 0
roi_update_interval = 5  # ROI 업데이트 간격 (프레임 단위)
prev_danger_roi = None
prev_warning_roi = None



# --------------------------- 함수 설정 ------------===----------------
# ------------------------ 높이 기반 거리 설정 ------------------------
def calculate_distance_from_height(vehicle_screen_height, class_id):
    if vehicle_screen_height == 0:
        return 0
    if class_id == 0:    # vehicle
        real_height = 2.0
    elif class_id == 1:  # big vehicle
        real_height = (2.2 if vehicle_screen_height < 50 else 3.0 
                           if vehicle_screen_height < 90 else 4.0)
    else:
        return 0
    camera_based_distance = (focal_length * real_height) / vehicle_screen_height
    return max(camera_based_distance - CAMERA_TO_BUMPER_OFFSET, 0)

# --------------------- 픽셀 기반 영역내 침범 탐지 ---------------------
def inside_roi(box, mask, threshold):
    x1, y1, x2, y2 = map(int, box)
    roi_box = mask[y1:y2, x1:x2]
    if roi_box.size == 0:
        return False
    inside = np.count_nonzero(roi_box == 255)
    return inside / roi_box.size >= threshold


# ------------------------ 그림자 제거 ------------------------
"""
그림자 제거용 HSV 채널 필터링 함수
cv2.inRange(gray or hsv, lower, upper) 특정 범위 색상만 필터링하여 마스크 생성

bitwise_and: 두 영상에서 공통된 부분만 남김 (예: 마스크 적용)
bitwise_not: 마스크 반전 → 그림자 제거 시 사용
ㄴ그림자인 부분이 1(255) -> bitwise_not으로 0으로
  그럼 bitwise_and는 그림자가 아닌 부분이 1 즉, 그림자 제거
"""
def remove_shadows_color_based(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([179, 19, 68])
    shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
    result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(shadow_mask))
    return result

# ------------------------ 사다리꼴 ROI 설정 ------------------------
"""
고정 값의 사다리꼴로 ROI 생성 후 <- 연산량, 노이즈(하늘, 보닛, 도로 외 정보) ↓
ROI 내부에서 차선을 추출하여
차선을 기준으로 ROI를 다시 생성
"""
def create_trapezoid_roi(frame, y_bottom, y_top):
    global prev_edges # 전역변수로 설정해 프레임 간 상태 공유 -> 함수 호출 끝날 때 사라짐 방지
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) # 히스토그램 평활화 : 명암 대비 향상 -> grayscale에만 적용 가능
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # 노이즈 제거
    edges = cv2.Canny(blur, 70, 140) # 엣지 검출 -> hist, blur 처리 후 사용할 것

    # 이전 에지 맵과 가중 평균하여 안정화
    if prev_edges is not None:
        edges = cv2.addWeighted(edges.astype(np.float32), 0.7, 
                                prev_edges.astype(np.float32), 0.3, 0).astype(np.uint8)
    prev_edges = edges.copy()

    """
    관심영역 설정
    프레임 픽셀과 일치시켜 영역 설정하면 객체 인식, 차선 검출 정확도↑
    또한, 해상도 변경 시 변경 픽셀값 그대로 적용됨
    """
    mask = np.zeros_like(edges) # 빈 마스크
    roi_vertices = np.array([[
        (width * 0.1, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255) # 영역에 흰색(255) 설정

    """
    ROI 내 차선 검출
    cv2.HoughLinesP(img, rho, theta, threshold, minLineLength, maxLineGap)
                (img, 거리 해상도, 각도 해상도, 직선 인식용 최소 점 수, 선 인식 최소길이, 이어지는 최대 간격)
    """
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=70)
    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # x2 = x1이 되면 직선이 되버리니 +1e-6한 것.
            if slope < -0.5: left_lines.append((x1, y1, x2, y2)) # 차량 기준으로 왼쪽 오른쪽 선 그린 것
            elif slope > 0.5: right_lines.append((x1, y1, x2, y2))

    # 여러 선분의 평균 직선 계산 → 좌우 차선 대표선 생성
    def average_line(lines):
        if not lines: return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]; y += [y1, y2] # 	여러 선분들의 양 끝점 좌표를 리스트에 누적
        return np.polyfit(y, x, 1) # 다항 회귀 : (y, x, 1)은 x를 y에 대해 1차 다항식(직선)으로 근사

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)
    if left_fit is None or right_fit is None: return None

    lx1, lx2 = int(np.polyval(left_fit, y_bottom)), int(np.polyval(left_fit, y_top))
    rx1, rx2 = int(np.polyval(right_fit, y_bottom)), int(np.polyval(right_fit, y_top))
    return np.array([[(lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)]], dtype=np.int32)


# --------------------------- 트랙바 설정 ---------------------------

cv2.namedWindow('YOLOv8 ROI Detection')
cv2.createTrackbar('Video Position', 'YOLOv8 ROI Detection', 0, 
                   video_length - 1, lambda val: cap.set(cv2.CAP_PROP_POS_FRAMES, val))
# cv2.createTrackbar(name, window, value, max, callback) -> cv2.namedWindow 있어야 작동함.

# --------------------------- 메인 루프 ---------------------------

while cap.isOpened():
    ret, frame = cap.read() # cap.read() 프레임 단위로 동영상 or 스트리밍 읽음
# ret : 읽기 성공 여부(True or False)
# frame : (1장의 이미지, numpy.ndarray) -> FPS = 30는 30[images]/1[sec]를 의미
    if not ret:
        break
    frame = cv2.resize(frame, (resize_width, resize_height))
    # cv2.resize(image, (width, height))
    
    # ROI 갱신 (간격마다)
    if frame_count % roi_update_interval == 0:
        danger_roi = create_trapezoid_roi(frame, danger_bottom, danger_top)
        warning_roi = create_trapezoid_roi(frame, warning_bottom, warning_top)
        if danger_roi is not None:
            prev_danger_roi = danger_roi
        if warning_roi is not None:
            prev_warning_roi = warning_roi
    else:
        danger_roi = prev_danger_roi
        warning_roi = prev_warning_roi

    # ROI 마스크 생성
    mask_danger = np.zeros((resize_height, resize_width), dtype=np.uint8)
    mask_warning = np.zeros((resize_height, resize_width), dtype=np.uint8)
    if danger_roi is not None:
        cv2.fillPoly(mask_danger, [danger_roi], 255)
        #검출된 ROI 영역을 마스크(mask_danger, mask_warning)로 만들어 사용
    if warning_roi is not None:
        cv2.fillPoly(mask_warning, [warning_roi], 255)
        mask_warning = cv2.subtract(mask_warning, mask_danger)  # 위험 ROI 제외

    # 시각화용 프레임 복사
    roi_overlay = frame.copy()
    frame_output = frame.copy()

    # ROI 시각화 (빨강/노랑 폴리라인)
    if danger_roi is not None:
        cv2.polylines(roi_overlay, [danger_roi], isClosed=True, color=(0, 0, 255), thickness=3)
        #검출된 ROI 영역을 마스크(mask_danger, mask_warning)로 만들어 사용
    if warning_roi is not None:
        warning_mask = np.zeros((resize_height, resize_width), dtype=np.uint8)
        cv2.fillPoly(warning_mask, [warning_roi], 255)
        danger_mask = np.zeros((resize_height, resize_width), dtype=np.uint8)
        cv2.fillPoly(danger_mask, [danger_roi], 255)
        warning_mask = cv2.subtract(warning_mask, danger_mask)
        contours, _ = cv2.findContours(warning_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3)

    # 객체 감지 수행 (ROI 내부만 표시)
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # 객체별 처리
    for box, class_id in zip(boxes, classes):
        class_id = int(class_id)
        if class_id not in class_names:
            continue
        x1, y1, x2, y2 = [int(c) for c in box]
        class_name = class_names[class_id]
        in_danger = inside_roi(box, mask_danger, danger_threshold)
        in_warning = inside_roi(box, mask_warning, warning_threshold)

        if in_danger or in_warning:
            color = (0, 0, 255) if in_danger else (0, 255, 255)
            cv2.rectangle(frame_output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_output, class_name, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if class_id in [0, 1]:
                distance = calculate_distance_from_height(y2 - y1, class_id)
                cv2.putText(frame_output, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            continue  # ROI 바깥 객체는 그리지 않음

    # FPS 계산 및 출력
    new_frame_time = time.time()
    fps_value = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame_output, f"FPS: {fps_value:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 오버레이 합성 후 디스플레이
    # frame_output(객체) + roi_overlay(ROI)를 함께 디스플레이
    # addWeighted() : ROI영역을 반투명으로 오버레이하는 비율(1:0.3)
    final_display = cv2.addWeighted(frame_output, 1.0, roi_overlay, 0.3, 0)
    cv2.imshow("YOLOv8 ROI Detection", final_display)

    # 키 입력 처리
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == 81:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 5000)
    elif key == 83:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 5000)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()