
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLOv8 모델 로드 및 초기화
model = YOLO('/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt')
model.eval()    # 모델을 테스트 모드로 추론 (학습용 x)
model.fuse()    # 모델 최적화 ->속도 향상 -> CPU일 때 향상률 ↑
_ = model(np.zeros((360, 640, 3), dtype=np.uint8))  # 워밍업

# 동영상 로드
cap = cv2.VideoCapture('/home/hkit/Downloads/Driving Downtown Seoul.mp4') # 동영상 열기
resize_width, resize_height = 640, 360  # 사이즈 변환
fps = cap.get(cv2.CAP_PROP_FPS) # 동영상 fps값 추출 (fps = 24)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상 전체 프레임 수
delay = int(1000 // (fps * 8))  # (fps*8)로 동영상 속도 개선

# 클래스 이름 정의
class_names = {
    0: "vehicle", 1: "big vehicle", 4: "bike",
    5: "human", 6: "animals", 7: "obstacles"
}

# 전역 변수로 이전 프레임의 차선 정보를 저장
prev_left_fit_global = None
prev_right_fit_global = None




# --- START: CAMERA-SPECIFIC FOCAL LENGTH CALCULATION ---
# Sony A7M4 + 35mm F1.4 GM specific parameters
# Sensor width for Sony A7M4 (full-frame) in mm
camera_sensor_width_mm = 35.9 # Approximate width of full-frame sensor
lens_focal_length_mm = 35.0 # Focal length of the Sony 35mm F1.4 GM lens

# Calculate focal length in pixels based on the resized image width
# This assumes the lens focal length is for the horizontal field of view
focal_length_pixels = (lens_focal_length_mm * resize_width) / camera_sensor_width_mm
# Using a more descriptive variable name
focal_length = focal_length_pixels
print(f"Calculated focal_length for Sony A7M4 35mm lens at {resize_width}x{resize_height}: {focal_length:.2f} pixels")
# --- END: CAMERA-SPECIFIC FOCAL LENGTH CALCULATION ---

# 차량의 실제 크기 (미터 단위) - vehicle과 big vehicle 구분
# 일반적인 차량의 실제 너비를 고려하여 값 조정 (예시)
# 승용차 대략 1.7m ~ 2.0m, 대형 차량 2.5m 이상
vehicle_real_length = 1.8  # 일반 차량 너비 (미터) - 이전 1.5에서 조정
big_vehicle_real_length = 2.5  # 대형 차량 너비 (미터) - 이전 4.0에서 조정

# 박스가 ROI 안에 일정 비율 이상 들어갔는지 확인하는 함수
def inside_roi(box, mask, threshold):   
    x1, y1, x2, y2 = map(int, box)
    roi_box = mask[y1:y2, x1:x2]
    if roi_box.size == 0:
        return False
    inside = np.count_nonzero(roi_box == 255)
    ratio = inside / roi_box.size
    return ratio >= threshold

# --- START: create_trapezoid_roi 함수 수정 ---
# 차선 기반으로 사다리꼴 ROI 생성
def create_trapezoid_roi(frame, y_bottom, y_top):
    height, width = frame.shape[:2]

    # 1. 영상 전처리: grayscale -> Gaussian Blur (커널 키움) -> CLAHE -> Canny Edge Detection (파라미터 튜닝)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur 커널 크기 증가 (노이즈 사전 제거 강화)
    blur = cv2.GaussianBlur(gray, (9, 9), 0) # (5,5)에서 (7,7)로 변경

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # clipLimit 미세 조정 (2.0에서 2.5로)
    clahe_output = clahe.apply(blur)

    # Canny Edge Detection 파라미터 튜닝 (노이즈 감소, 강한 에지만 추출)
    # 기존 (50, 150)에서 (70, 200)으로 변경 (이전 권장 사항 반영)
    # 필요시 (80, 220) 또는 (100, 250) 등으로 추가 튜닝
    edges = cv2.Canny(clahe_output, 80, 220)

    # 디버깅을 위한 중간 결과 이미지 표시 (주석 해제하여 확인)
    # cv2.imshow("Original Gray", gray)
    # cv2.imshow("Blurred Image", blur)
    cv2.imshow("CLAHE Output", clahe_output) # CLAHE 적용 후 이미지
    cv2.imshow("Canny Edges", edges)       # Canny 적용 후 에지 이미지

    # ROI 마스크 생성 (차선 검출 영역 제한)
    # 이 ROI는 Hough 변환이 차선을 찾는 영역을 제한하는 데 사용됩니다.
    mask = np.zeros_like(edges)
    # 초기 ROI 영역을 좀 더 넓게 설정하여 차선이 놓이는 범위에 유연성 확보
    # width // 2 - 60, width // 2 + 60 에서 변경
    roi_vertices = np.array([[
        (90, height),                    # Bottom-left (영상 하단 좌측 끝)
        (width // 2 - 50, height // 2 + 40), # Top-left (윗부분 좌측 끝점 조정)
        (width // 2 + 50, height // 2 + 40), # Top-right (윗부분 우측 끝점 조정)
        (width - 60, height)                 # Bottom-right (영상 하단 우측 끝)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask) # ROI 내의 에지만 남김

    cv2.imshow("Masked Edges for Hough", masked_edges) # Hough 변환 입력 이미지 확인

    # 허프 변환 - 왼쪽 기울기 <0, 오른쪽 기울기 >0로 직선을 분류
    # HoughLinesP 파라미터 튜닝
    # threshold, minLineLength, maxLineGap 조정 (비디오에 따라 미세 조정 필요)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 40, # threshold: 50 -> 40 (더 많은 선 검출 시도)
                            minLineLength=50, # minLineLength: 40 -> 50 (더 긴 선만 유효)
                            maxLineGap=150)    # maxLineGap: 100 -> 80 (선 끊김 허용 범위 조절)

    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            # 수직선에 대한 0으로 나누기 오류 방지 (x2 - x1이 매우 작을 때)
            slope = (y2 - y1) / (x2 - x1 + 1e-6) 
            # 차선의 기울기 범위를 조금 더 엄격하게 설정 (필요시 조절)
            if -1.0 < slope < -0.1: # 왼쪽 차선 기울기 범위 (-0.5에서 확장/조정)
                left_lines.append((x1, y1, x2, y2))
            elif 0.1 < slope < 1.0: # 오른쪽 차선 기울기 범위 (0.5에서 확장/조정)
                right_lines.append((x1, y1, x2, y2))




        # 디버깅: 검출된 선들을 영상에 그려서 확인
    debug_frame_lines = frame.copy() # 원본 프레임을 복사하여 그릴 것
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (0, 255, 255), 1) # 모든 허프 선 (노랑)
    if left_lines:
        for x1, y1, x2, y2 in left_lines:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (255, 0, 0), 2) # 왼쪽 차선으로 분류된 선 (파랑)
    if right_lines:
        for x1, y1, x2, y2 in right_lines:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (0, 0, 255), 2) # 오른쪽 차선으로 분류된 선 (빨강)
    cv2.imshow("Detected and Classified Hough Lines", debug_frame_lines)



    def average_line(lines):
        if not lines:
            return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]
            y += [y1, y2]
        # numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares 오류 방지
        # 최소 2개의 점이 있어야 polyfit이 가능
        if len(y) < 2:
            return None
        return np.polyfit(y, x, deg=1)  # x = ay + b

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)


    # 이전 프레임의 차선 정보를 활용하여 ROI 안정화 (간단한 이동 평균)
    global prev_left_fit_global, prev_right_fit_global # <<--- 이 줄 추가!    

    # 이전 프레임의 차선 정보를 활용하여 ROI 안정화 (간단한 이동 평균)
    # 이 기능을 활성화하려면 create_trapezoid_roi 함수 외부에 전역 변수 필요
    # 예: prev_left_fit_global, prev_right_fit_global
    # 여기서는 함수의 복잡도를 높이지 않기 위해 일단 주석 처리합니다.
    # 만약 ROI가 여전히 불안정하다면 이 로직을 추가하는 것을 고려하세요.
    # global prev_left_fit_global, prev_right_fit_global
    alpha = 0.7 # 새 값 반영 비율
    if left_fit is not None:
        if prev_left_fit_global is not None:
            left_fit = alpha * left_fit + (1 - alpha) * prev_left_fit_global
        prev_left_fit_global = left_fit
    if right_fit is not None:
        if prev_right_fit_global is not None:
            right_fit = alpha * right_fit + (1 - alpha) * prev_right_fit_global
        prev_right_fit_global = right_fit


    if left_fit is None or right_fit is None:
        # 두 차선 중 하나라도 감지되지 않으면 None 반환하여 ROI 생성 건너뛰기
        # 또는 이전 프레임의 ROI를 재활용하는 로직 추가 가능
        return None
    

    # ROI 꼭짓점 계산
    lx1 = int(np.polyval(left_fit, y_bottom))
    lx2 = int(np.polyval(left_fit, y_top))
    rx1 = int(np.polyval(right_fit, y_bottom))
    rx2 = int(np.polyval(right_fit, y_top))

   # 디버깅: 최종 피팅된 차선으로 ROI가 어떻게 생성되는지 확인
    # 이 부분은 ROI 꼭짓점 계산 바로 아래에 추가하세요.
    if left_fit is not None and right_fit is not None:
        debug_frame_roi_fit = frame.copy()
        cv2.line(debug_frame_roi_fit, (lx1, y_bottom), (lx2, y_top), (255, 0, 0), 2) # 파랑 (왼쪽 최종 차선)
        cv2.line(debug_frame_roi_fit, (rx1, y_bottom), (rx2, y_top), (0, 0, 255), 2) # 빨강 (오른쪽 최종 차선)
        cv2.imshow("Final Fitted Lines for ROI Calculation", debug_frame_roi_fit)




    # ROI의 x 좌표가 이미지 범위를 벗어나지 않도록 클리핑 (중요!)
    lx1 = np.clip(lx1, 0, width)
    lx2 = np.clip(lx2, 0, width)
    rx1 = np.clip(rx1, 0, width)
    rx2 = np.clip(rx2, 0, width)
    
    # 검출된 차선이 너무 가깝거나 교차하는 경우, ROI 생성을 피하거나 조정
    # (예: lx1 >= rx1 또는 lx2 >= rx2 인 경우)
    if lx1 >= rx1 or lx2 >= rx2:
        print("Warning: Detected lanes are crossing or too close for ROI creation. Skipping ROI.")
        return None


    # 디버깅: 피팅된 차선으로 ROI가 어떻게 생성되는지 확인 (옵션)
    # debug_frame_roi_fit = frame.copy()
    # cv2.line(debug_frame_roi_fit, (lx1, y_bottom), (lx2, y_top), (255, 0, 0), 2)
    # cv2.line(debug_frame_roi_fit, (rx1, y_bottom), (rx2, y_top), (0, 0, 255), 2)
    # cv2.imshow("Fitted Lines for ROI", debug_frame_roi_fit)


    return np.array([[
        (lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)
    ]], dtype=np.int32)
# --- END: create_trapezoid_roi 함수 수정 ---


# ROI 높이 설정 - 360은 하단값, 뒤의 값은 상단값으로 작을수록 영역이 길어짐.
danger_bottom, danger_top = 360, 300
warning_bottom, warning_top = 360, 260

# ROI 별 threshold 설정 - 픽셀 비율로 ROI 영역 침범 시 감지
danger_threshold = 0.2    
warning_threshold = 0.3

# 거리 계산 함수
def calculate_distance(vehicle_screen_width, vehicle_real_length):
    #차량의 화면 너비와 실제 차량 크기를 비교하여 거리 계산
    return (focal_length * vehicle_real_length) / vehicle_screen_width

# 트랙바 콜백 함수
def update_video_position(val):
    #트랙바 값에 따라 동영상 재생 위치를 업데이트
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)

# 트랙바 생성
cv2.namedWindow('YOLOv8 ROI Detection')
cv2.createTrackbar('Video Position', 'YOLOv8 ROI Detection', 0, video_length - 1, update_video_position)

# FPS 측정을 위한 변수
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():

    # 프레임 읽고 사이즈 조정
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resize_width, resize_height))

    # 빈 마스크 생성
    mask_danger = np.zeros((resize_height, resize_width), dtype=np.uint8)
    mask_warning = np.zeros((resize_height, resize_width), dtype=np.uint8)

    # danger ROI 생성
    danger_roi = create_trapezoid_roi(frame, danger_bottom, danger_top)
    if danger_roi is not None:
        cv2.fillPoly(mask_danger, [danger_roi], 255)

    # warning ROI 생성
    warning_roi = create_trapezoid_roi(frame, warning_bottom, warning_top)
    if warning_roi is not None:
        cv2.fillPoly(mask_warning, [warning_roi], 255)
        # danger ROI와 겹치는 부분 제외 (마스크 빼기)
        if danger_roi is not None: # danger_roi가 존재할 때만 뺀다
            danger_mask_for_subtract = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(danger_mask_for_subtract, [danger_roi], 255)
            mask_warning = cv2.subtract(mask_warning, danger_mask_for_subtract)


    roi_overlay = frame.copy()

    # Danger ROI 테두리: 무조건 먼저 빨간색으로 그림
    if danger_roi is not None:
        cv2.polylines(roi_overlay, [danger_roi], isClosed=True, color=(0, 0, 255), thickness=3)

    # Warning ROI 테두리: Danger ROI와 겹치는 부분은 제외하고 노란색으로 그림
    if warning_roi is not None:
        if danger_roi is not None:
            # Warning ROI 마스크 만들기
            warning_mask_draw = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(warning_mask_draw, [warning_roi], 255)

            # Danger ROI 마스크 빼기 (겹치는 부분 제거)
            danger_mask_draw = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(danger_mask_draw, [danger_roi], 255)

            # Warning 마스크에서 Danger 부분 제거
            warning_mask_draw = cv2.subtract(warning_mask_draw, danger_mask_draw)

            # 남은 Warning 부분의 외곽선 찾아 테두리 그림
            contours, _ = cv2.findContours(warning_mask_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3)
        else:
            # Danger ROI가 없으면 Warning ROI 전체를 그림
            cv2.polylines(roi_overlay, [warning_roi], isClosed=True, color=(0, 255, 255), thickness=3)

    # 객체 탐지
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, class_id in zip(boxes, classes):
        class_id = int(class_id)
        if class_id not in class_names:
            continue

        x1, y1, x2, y2 = [int(c) for c in box]
        class_name = class_names[class_id]

        # 객체 탐지 시 색 지정 - danger, warning 외
        if inside_roi(box, mask_danger, danger_threshold):
            color = (0, 0, 255) # 빨강 (Danger)
        elif inside_roi(box, mask_warning, warning_threshold):
            color = (0, 255, 255) # 노랑 (Warning)
        else:
            color = (0, 255, 0) # 초록 (정상)

        # 차량의 화면 상 너비 계산
        vehicle_screen_width = x2 - x1

        # 거리 계산 (차량에 대해서만 계산)
        if class_id == 0 or class_id == 1:  # big vehicle과 vehicle에 대해서만 거리 계산
            vehicle_real_length_used = vehicle_real_length if class_id == 0 else big_vehicle_real_length 
            
            # 픽셀 너비가 0이 되는 경우 방지
            if vehicle_screen_width > 0:
                distance = calculate_distance(vehicle_screen_width, vehicle_real_length_used)
            else:
                distance = float('inf') # 화면 너비가 0이면 거리를 무한대로 설정

            # 거리 표시
            cv2.putText(frame, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 바운딩 박스, 클래스별 라벨 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS 계산
    new_frame_time = time.time()  # 현재 시간을 프레임 처리 시간으로 저장
    fps_value = 1 / (new_frame_time - prev_frame_time)  # FPS 계산
    prev_frame_time = new_frame_time  # 이전 프레임 시간 업데이트

    # FPS 출력
    cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 표시
    overlay = cv2.addWeighted(frame, 1.0, roi_overlay, 0.3, 0)
    cv2.imshow("YOLOv8 ROI Detection", overlay)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == 81: # 'Q' 키 (대문자)
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 5000)
    elif key == 83: # 'S' 키 (대문자)
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 5000)

cap.release()
cv2.destroyAllWindows()
"""




import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLOv8 모델 로드 및 초기화
model = YOLO('/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt')
model.eval()    # 모델을 테스트 모드로 추론 (학습용 x)
model.fuse()    # 모델 최적화 ->속도 향상 -> CPU일 때 향상률 ↑
_ = model(np.zeros((360, 640, 3), dtype=np.uint8))  # 워밍업

# 동영상 로드
cap = cv2.VideoCapture('/home/hkit/Downloads/Driving Downtown Seoul.mp4') # 동영상 열기
resize_width, resize_height = 640, 360  # 사이즈 변환
fps = cap.get(cv2.CAP_PROP_FPS) # 동영상 fps값 추출 (fps = 24)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상 전체 프레임 수
delay = int(1000 // (fps * 8))  # (fps*8)로 동영상 속도 개선

# 클래스 이름 정의
class_names = {
    0: "vehicle", 1: "big vehicle", 4: "bike",
    5: "human", 6: "animals", 7: "obstacles"
}

# 전역 변수로 이전 프레임의 차선 정보를 저장 (이동 평균용)
prev_left_fit_global = None
prev_right_fit_global = None

# <<< 추가: 화면에 표시될 마지막 유효한 ROI를 저장할 전역 변수 >>>
last_display_danger_roi = None
last_display_warning_roi = None


# --- START: CAMERA-SPECIFIC FOCAL LENGTH CALCULATION ---
# Sony A7M4 + 35mm F1.4 GM specific parameters
# Sensor width for Sony A7M4 (full-frame) in mm
camera_sensor_width_mm = 35.9 # Approximate width of full-frame sensor
lens_focal_length_mm = 35.0 # Focal length of the Sony 35mm F1.4 GM lens

# Calculate focal length in pixels based on the resized image width
# This assumes the lens focal length is for the horizontal field of view
focal_length_pixels = (lens_focal_length_mm * resize_width) / camera_sensor_width_mm
# Using a more descriptive variable name
focal_length = focal_length_pixels
print(f"Calculated focal_length for Sony A7M4 35mm lens at {resize_width}x{resize_height}: {focal_length:.2f} pixels")
# --- END: CAMERA-SPECIFIC FOCAL LENGTH CALCULATION ---

# 차량의 실제 크기 (미터 단위) - vehicle과 big vehicle 구분
# 일반적인 차량의 실제 너비를 고려하여 값 조정 (예시)
# 승용차 대략 1.7m ~ 2.0m, 대형 차량 2.5m 이상
vehicle_real_length = 1.8  # 일반 차량 너비 (미터) - 이전 1.5에서 조정
big_vehicle_real_length = 2.5  # 대형 차량 너비 (미터) - 이전 4.0에서 조정

# 박스가 ROI 안에 일정 비율 이상 들어갔는지 확인하는 함수
def inside_roi(box, mask, threshold):   
    x1, y1, x2, y2 = map(int, box)
    roi_box = mask[y1:y2, x1:x2]
    if roi_box.size == 0:
        return False
    inside = np.count_nonzero(roi_box == 255)
    ratio = inside / roi_box.size
    return ratio >= threshold

# --- START: create_trapezoid_roi 함수 수정 ---
# 차선 기반으로 사다리꼴 ROI 생성
def create_trapezoid_roi(frame, y_bottom, y_top):
    height, width = frame.shape[:2]

    # 1. 영상 전처리: grayscale -> Gaussian Blur (커널 키움) -> CLAHE -> Canny Edge Detection (파라미터 튜닝)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur 커널 크기 증가 (노이즈 사전 제거 강화)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_output = clahe.apply(blur)

    # Canny Edge Detection 파라미터 튜닝 (노이즈 감소, 강한 에지만 추출)
    edges = cv2.Canny(clahe_output, 80, 200)

    # 디버깅을 위한 중간 결과 이미지 표시 (주석 해제하여 확인)
    # cv2.imshow("Original Gray", gray)
    # cv2.imshow("Blurred Image", blur)
    cv2.imshow("CLAHE Output", clahe_output)
    cv2.imshow("Canny Edges", edges)

    # ROI 마스크 생성 (차선 검출 영역 제한)
    # 이 ROI는 Hough 변환이 차선을 찾는 영역을 제한하는 데 사용됩니다.
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (90, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width - 60, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask) # ROI 내의 에지만 남김

    cv2.imshow("Masked Edges for Hough", masked_edges) # Hough 변환 입력 이미지 확인

    # 허프 변환 - 왼쪽 기울기 <0, 오른쪽 기울기 >0로 직선을 분류
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 40,
                            minLineLength=60,
                            maxLineGap=150) # maxLineGap 150으로 조정

    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6) 
            if -1.0 < slope < -0.1:
                left_lines.append((x1, y1, x2, y2))
            elif 0.1 < slope < 1.0:
                right_lines.append((x1, y1, x2, y2))

    # 디버깅: 검출된 선들을 영상에 그려서 확인
    debug_frame_lines = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (0, 255, 255), 1)
    if left_lines:
        for x1, y1, x2, y2 in left_lines:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if right_lines:
        for x1, y1, x2, y2 in right_lines:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Detected and Classified Hough Lines", debug_frame_lines)


    def average_line(lines):
        if not lines:
            return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]
            y += [y1, y2]
        if len(y) < 2:
            return None
        return np.polyfit(y, x, deg=1)

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)

    # 이전 프레임의 차선 정보를 활용하여 ROI 안정화 (간단한 이동 평균)
    global prev_left_fit_global, prev_right_fit_global # <<--- 이 줄!    
    alpha = 0.85 # alpha 값 0.7로 조정
    if left_fit is not None:
        if prev_left_fit_global is not None:
            left_fit = alpha * left_fit + (1 - alpha) * prev_left_fit_global
        prev_left_fit_global = left_fit
    if right_fit is not None:
        if prev_right_fit_global is not None:
            right_fit = alpha * right_fit + (1 - alpha) * prev_right_fit_global
        prev_right_fit_global = right_fit

    if left_fit is None or right_fit is None:
        return None
    
    lx1 = int(np.polyval(left_fit, y_bottom))
    lx2 = int(np.polyval(left_fit, y_top))
    rx1 = int(np.polyval(right_fit, y_bottom))
    rx2 = int(np.polyval(right_fit, y_top))

    # 디버깅: 최종 피팅된 차선으로 ROI가 어떻게 생성되는지 확인
    debug_frame_roi_fit = frame.copy()
    cv2.line(debug_frame_roi_fit, (lx1, y_bottom), (lx2, y_top), (255, 0, 0), 2)
    cv2.line(debug_frame_roi_fit, (rx1, y_bottom), (rx2, y_top), (0, 0, 255), 2)
    cv2.imshow("Final Fitted Lines for ROI Calculation", debug_frame_roi_fit)

    lx1 = np.clip(lx1, 0, width)
    lx2 = np.clip(lx2, 0, width)
    rx1 = np.clip(rx1, 0, width)
    rx2 = np.clip(rx2, 0, width)
    
    if lx1 >= rx1 or lx2 >= rx2:
        print("Warning: Detected lanes are crossing or too close for ROI creation. Skipping ROI.")
        return None

    return np.array([[
        (lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)
    ]], dtype=np.int32)
# --- END: create_trapezoid_roi 함수 수정 ---


# ROI 높이 설정
danger_bottom, danger_top = 360, 300
warning_bottom, warning_top = 360, 260

# ROI 별 threshold 설정
danger_threshold = 0.2    
warning_threshold = 0.3

# 거리 계산 함수
def calculate_distance(vehicle_screen_width, vehicle_real_length):
    return (focal_length * vehicle_real_length) / vehicle_screen_width

# 트랙바 콜백 함수
def update_video_position(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)

# 트랙바 생성
cv2.namedWindow('YOLOv8 ROI Detection')
cv2.createTrackbar('Video Position', 'YOLOv8 ROI Detection', 0, video_length - 1, update_video_position)

# FPS 측정을 위한 변수
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (resize_width, resize_height))

    # 빈 마스크 생성 (객체 감지 필터링용 - 현재 프레임의 ROI 사용)
    mask_danger = np.zeros((resize_height, resize_width), dtype=np.uint8)
    mask_warning = np.zeros((resize_height, resize_width), dtype=np.uint8)

    # danger ROI 생성 (현재 프레임의 ROI)
    current_danger_roi = create_trapezoid_roi(frame, danger_bottom, danger_top)
    if current_danger_roi is not None:
        cv2.fillPoly(mask_danger, [current_danger_roi], 255)
        last_display_danger_roi = current_danger_roi # <<< 수정: 마지막 유효 ROI 업데이트

    # warning ROI 생성 (현재 프레임의 ROI)
    current_warning_roi = create_trapezoid_roi(frame, warning_bottom, warning_top)
    if current_warning_roi is not None:
        cv2.fillPoly(mask_warning, [current_warning_roi], 255)
        # danger ROI와 겹치는 부분 제외 (마스크 빼기)
        if current_danger_roi is not None:
            danger_mask_for_subtract = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(danger_mask_for_subtract, [current_danger_roi], 255)
            mask_warning = cv2.subtract(mask_warning, danger_mask_for_subtract)
        last_display_warning_roi = current_warning_roi # <<< 수정: 마지막 유효 ROI 업데이트
    elif current_danger_roi is not None and last_display_warning_roi is not None:
        # Warning ROI 마스크를 Danger ROI 마스크가 겹치지 않게 만듦 (이전 Warning ROI를 기반으로)
        # 이 로직은 inside_roi 함수에서 사용되는 마스크를 조정하는 부분입니다.
        danger_mask_for_subtract = np.zeros((resize_height, resize_width), dtype=np.uint8)
        cv2.fillPoly(danger_mask_for_subtract, [current_danger_roi], 255)
        
        temp_warning_mask = np.zeros((resize_height, resize_width), dtype=np.uint8)
        # 주의: 이 시점의 mask_warning은 이미 current_warning_roi (None일 수 있음)로 초기화되어 있으므로,
        # 여기서는 last_display_warning_roi를 사용해 일시 마스크를 만들고 danger를 는 방식이 더 안전합니다.
        # 하지만 inside_roi는 이 마스크를 사용하므로, 만약 current_warning_roi가 None이면 이 mask_warning은 비어있어야 합니다.
        # 아래 로직은 `current_warning_roi`가 None일 때, `last_display_warning_roi`가 있다고 해도,
        # 실제로 warning ROI에 대한 탐지를 비활성화하고 싶다면 수정이 필요합니다.
        # 현재는 이전에 정의된 `mask_warning`이 비어있을 가능성이 높습니다.
        # 시각화와 탐지 마스크를 분리하는 것이 핵심입니다.
        pass # 마스크에 대한 로직은 그대로 유지하고, 시각화만 last_display_roi를 사용합니다.


    roi_overlay = frame.copy()

    # <<< 수정: ROI 그리기 로직 - last_display_roi 사용 >>>
    # Danger ROI 테두리: last_display_danger_roi가 있으면 그림
    if last_display_danger_roi is not None:
        cv2.polylines(roi_overlay, [last_display_danger_roi], isClosed=True, color=(0, 0, 255), thickness=3)

    # Warning ROI 테두리: last_display_warning_roi가 있으면 그림
    if last_display_warning_roi is not None:
        # Danger ROI와 겹치는 부분 제외 로직은 여전히 필요 (시각적 일관성을 위해)
        if last_display_danger_roi is not None:
            warning_mask_draw = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(warning_mask_draw, [last_display_warning_roi], 255) # last_display_warning_roi 사용

            danger_mask_draw = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(danger_mask_draw, [last_display_danger_roi], 255) # last_display_danger_roi 사용

            warning_mask_draw = cv2.subtract(warning_mask_draw, danger_mask_draw)

            contours, _ = cv2.findContours(warning_mask_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3)
        else:
            # Danger ROI가 없으면 Warning ROI 전체를 그림
            cv2.polylines(roi_overlay, [last_display_warning_roi], isClosed=True, color=(0, 255, 255), thickness=3)
    # <<< 수정 끝 >>>

    # 객체 탐지
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, class_id in zip(boxes, classes):
        class_id = int(class_id)
        if class_id not in class_names:
            continue

        x1, y1, x2, y2 = [int(c) for c in box]
        class_name = class_names[class_id]

        # 객체 탐지 시 색 지정 - danger, warning 외
        # 여기서는 current_danger_roi와 current_warning_roi를 사용하는 mask_danger, mask_warning을 사용하는 것이 맞습니다.
        # (현재 차선이 감지된 경우에만 경고를 발생시켜야 하므로)
        if inside_roi(box, mask_danger, danger_threshold): # mask_danger는 current_danger_roi 기반
            color = (0, 0, 255) # 빨강 (Danger)
        elif inside_roi(box, mask_warning, warning_threshold): # mask_warning은 current_warning_roi 기반
            color = (0, 255, 255) # 노랑 (Warning)
        else:
            color = (0, 255, 0) # 초록 (정상)

        # 차량의 화면 상 너비 계산
        vehicle_screen_width = x2 - x1

        # 거리 계산 (차량에 대해서만 계산)
        if class_id == 0 or class_id == 1:
            vehicle_real_length_used = vehicle_real_length if class_id == 0 else big_vehicle_real_length 
            
            if vehicle_screen_width > 0:
                distance = calculate_distance(vehicle_screen_width, vehicle_real_length_used)
            else:
                distance = float('inf')

            cv2.putText(frame, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS 계산
    new_frame_time = time.time()
    fps_value = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 표시
    overlay = cv2.addWeighted(frame, 1.0, roi_overlay, 0.3, 0)
    cv2.imshow("YOLOv8 ROI Detection", overlay)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == 81: # 'Q' 키 (대문자)
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 5000)
    elif key == 83: # 'S' 키 (대문자)
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 5000)

cap.release()
cv2.destroyAllWindows()