
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- 1. YOLOv8 모델 로드 및 초기화 ---
# 모델 파일 경로를 당신의 실제 경로로 설정하세요.
model = YOLO('/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt')
model.eval()    # 모델을 추론(테스트) 모드로 설정합니다.
model.fuse()    # 모델 최적화 (CPU 사용 시 속도 향상에 도움).
_ = model(np.zeros((360, 640, 3), dtype=np.uint8))  # 모델 워밍업 (첫 추론 시간 단축).

# --- 2. 동영상 로드 및 초기 설정 ---
# 동영상 파일 경로를 당신의 실제 경로로 설정하세요.
cap = cv2.VideoCapture('/home/hkit/Downloads/test_movie_009.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit() # 동영상 파일을 열 수 없으면 프로그램 종료

resize_width, resize_height = 640, 360  # 프레임 리사이즈 해상도.
fps = cap.get(cv2.CAP_PROP_FPS) # 동영상의 원본 FPS 추출.
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상 전체 프레임 수.
delay = int(1000 // (fps * 8))  # 동영상 재생 속도 조절 (원본 FPS의 8배 빠르게).

# --- 3. 클래스 이름 정의 (YOLOv8 모델의 클래스에 맞게) ---
class_names = {
    0: "vehicle", 1: "big vehicle", 4: "bike",
    5: "human", 6: "animals", 7: "obstacles"
}

# --- 4. 전역 변수 초기화 ---
# 이전 프레임의 차선 정보를 저장하여 이동 평균(smoothing)에 사용합니다.
prev_left_fit_global = None
prev_right_fit_global = None

# 화면에 표시될 마지막 유효한 ROI(관심 영역)를 저장합니다.
# 차선 인식이 일시적으로 실패해도 ROI가 화면에서 사라지지 않도록 합니다.
last_display_danger_roi = None
last_display_warning_roi = None

# --- 5. 카메라 관련 초점 거리 계산 (거리 추정용) ---
# Sony A7M4 + 35mm F1.4 GM 렌즈 사양 예시.
camera_sensor_width_mm = 35.9 # 풀프레임 센서의 대략적인 너비 (mm).
lens_focal_length_mm = 35.0 # 렌즈의 초점 거리 (mm).

# 리사이즈된 이미지 너비에 기반한 픽셀 단위 초점 거리 계산.
# 이는 렌즈 초점 거리가 수평 시야에 대한 것이라고 가정합니다.
focal_length = (lens_focal_length_mm * resize_width) / camera_sensor_width_mm
print(f"Calculated focal_length for Sony A7M4 35mm lens at {resize_width}x{resize_height}: {focal_length:.2f} 픽셀")

# --- 6. 차량의 실제 크기 정의 (미터 단위) ---
vehicle_real_length = 1.8  # 일반 차량의 실제 너비 (미터).
big_vehicle_real_length = 2.5  # 대형 차량의 실제 너비 (미터).

# --- 7. 유틸리티 함수: 박스가 ROI 안에 일정 비율 이상 들어갔는지 확인 ---
def inside_roi(box, mask, threshold):
    x1, y1, x2, y2 = map(int, box)
    # 박스 영역이 마스크 범위를 벗어나지 않도록 클리핑.
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(mask.shape[0], y2)
    x2 = min(mask.shape[1], x2)

    roi_box_region = mask[y1:y2, x1:x2]
    if roi_box_region.size == 0: # 유효하지 않은 박스 영역일 경우.
        return False
    # ROI 마스크에서 255 (ROI 영역)인 픽셀 수 계산.
    inside_pixels = np.count_nonzero(roi_box_region == 255)
    ratio = inside_pixels / roi_box_region.size
    return ratio >= threshold

# --- 8. 유틸리티 함수: 차선 기반 사다리꼴 ROI 생성 ---
def create_trapezoid_roi(frame, y_bottom, y_top):
    height, width = frame.shape[:2]

    # 1. 영상 전처리: grayscale -> Gaussian Blur (커널 키움) -> CLAHE -> Canny Edge Detection (파라미터 튜닝)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # --- 디버깅: 그레이스케일 이미지 확인 ---
    # cv2.imshow("Debug_1_Grayscale", gray)
    
    blur = cv2.GaussianBlur(gray, (9, 9), 0) # Gaussian Blur 커널 크기 증가 (노이즈 제거 강화).
    # --- 디버깅: 블러 처리된 이미지 확인 ---
    # cv2.imshow("Debug_2_Blurred", blur)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_output = clahe.apply(blur) # CLAHE 적용 (국부 대비 향상).
    # --- 디버깅: CLAHE 적용 이미지 확인 ---
    # cv2.imshow("Debug_3_CLAHE_Output", clahe_output)

    # Canny 엣지 감지 (파라미터 튜닝: 노이즈 감소, 강한 엣지만 추출).
    edges = cv2.Canny(clahe_output, 80, 200)
    # --- 디버깅: Canny 엣지 이미지 확인 ---
    cv2.imshow("Debug_4_Canny_Edges", edges)

    # 2. 차선 감지를 위한 ROI 마스크 생성:
    # 이 ROI는 Hough 변환이 차선을 찾는 영역을 제한하는 데 사용됩니다.
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (90, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width - 60, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask) # ROI 내의 엣지만 남김.

    # --- 디버깅: Hough 변환 입력 이미지 (ROI 마스킹된 엣지) 확인 ---
    cv2.imshow("Debug_5_Masked_Edges_for_Hough", masked_edges)

    # 3. 허프 변환을 통한 선 감지 및 분류:
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 40,
                            minLineLength=60,
                            maxLineGap=150) # maxLineGap 조정.

    left_lines, right_lines = [], []
    # --- 가로선 무시 로직 및 기울기 기반 분류 ---
    slope_threshold_min = 0.1 # 이 값보다 기울기 절대값이 작으면 수평선에 가깝다고 판단하여 무시.
    slope_threshold_max = 10.0 # 이 값보다 기울기 절대값이 크면 너무 수직인 선으로 판단하여 무시.

    if lines is not None:
        # --- 디버깅: 감지된 전체 선의 개수 출력 ---
        # print(f"Debug: HoughLinesP detected {len(lines)} raw lines.")
        for x1, y1, x2, y2 in lines[:, 0]:
            # 수직선 방지 및 0으로 나누는 것 방지.
            if x2 - x1 == 0:
                slope = 999.0 # 매우 큰 값으로 설정하여 수직선으로 처리 (추후 필터링됨).
            else:
                slope = (y2 - y1) / (x2 - x1)
            
            # 기울기 필터링: 일정 기울기 이상 (수평선 무시) 및 너무 수직인 선 무시.
            if abs(slope) < slope_threshold_min or abs(slope) > slope_threshold_max:
                continue # 이 범위의 기울기는 차선으로 간주하지 않음.

            if slope < -slope_threshold_min: # 왼쪽 차선 (음수 기울기).
                left_lines.append((x1, y1, x2, y2))
            elif slope > slope_threshold_min: # 오른쪽 차선 (양수 기울기).
                right_lines.append((x1, y1, x2, y2))
    
    # --- 디버깅: 분류된 왼쪽/오른쪽 차선 개수 출력 ---
    # print(f"Debug: Left lines after filtering: {len(left_lines)}, Right lines after filtering: {len(right_lines)}")

    # --- 디버깅: 검출 및 분류된 선들을 영상에 그려서 확인 ---
    debug_frame_lines = frame.copy()
    if left_lines:
        for x1, y1, x2, y2 in left_lines:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (255, 0, 0), 2) # 파란색 (왼쪽 차선).
    if right_lines:
        for x1, y1, x2, y2 in right_lines:
            cv2.line(debug_frame_lines, (x1, y1), (x2, y2), (0, 0, 255), 2) # 빨간색 (오른쪽 차선).
    cv2.imshow("Debug_6_Detected_and_Classified_Hough_Lines", debug_frame_lines)


    # 4. 선들을 평균화하여 하나의 대표선으로 피팅:
    def average_line(lines):
        if not lines:
            return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]
            y += [y1, y2]
        if len(y) < 2: # 최소 2개 점이 있어야 피팅 가능.
            return None
        return np.polyfit(y, x, deg=1) # 1차 함수로 피팅 (x = my + b 형태).

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)
    
    # --- 디버깅: 피팅된 기울기/절편 값 출력 ---
    # if left_fit is not None:
    #     print(f"Debug: Left fit coefficients (slope, intercept): {left_fit}")
    # if right_fit is not None:
    #     print(f"Debug: Right fit coefficients (slope, intercept): {right_fit}")

    # 5. 이전 프레임의 차선 정보를 활용하여 ROI 안정화 (이동 평균):
    global prev_left_fit_global, prev_right_fit_global
    alpha = 0.85 # 이동 평균 가중치 (현재 프레임에 0.85, 이전 프레임에 0.15).
    if left_fit is not None:
        if prev_left_fit_global is not None:
            left_fit = alpha * left_fit + (1 - alpha) * prev_left_fit_global
        prev_left_fit_global = left_fit
    if right_fit is not None:
        if prev_right_fit_global is not None:
            right_fit = alpha * right_fit + (1 - alpha) * prev_right_fit_global
        prev_right_fit_global = right_fit

    if left_fit is None or right_fit is None:
        # --- 디버깅: 차선 피팅 실패 시 메시지 출력 ---
        # print("Debug: Could not fit both left and right lanes. Returning None for ROI.")
        return None
    
    # 6. 피팅된 선을 기준으로 ROI의 꼭짓점 계산:
    lx1 = int(np.polyval(left_fit, y_bottom)) # 왼쪽 차선의 y_bottom 높이에서의 x 좌표.
    lx2 = int(np.polyval(left_fit, y_top))    # 왼쪽 차선의 y_top 높이에서의 x 좌표.
    rx1 = int(np.polyval(right_fit, y_bottom)) # 오른쪽 차선의 y_bottom 높이에서의 x 좌표.
    rx2 = int(np.polyval(right_fit, y_top))    # 오른쪽 차선의 y_top 높이에서의 x 좌표.

    # --- 디버깅: 최종 피팅된 차선으로 ROI가 어떻게 생성되는지 확인 ---
    debug_frame_roi_fit = frame.copy()
    cv2.line(debug_frame_roi_fit, (lx1, y_bottom), (lx2, y_top), (255, 0, 0), 2) # 파란색 (왼쪽).
    cv2.line(debug_frame_roi_fit, (rx1, y_bottom), (rx2, y_top), (0, 0, 255), 2) # 빨간색 (오른쪽).
    # cv2.imshow("Debug_7_Final_Fitted_Lines_for_ROI", debug_frame_roi_fit)

    # 7. ROI 꼭짓점의 x 좌표가 이미지 범위를 벗어나지 않도록 클리핑:
    lx1 = np.clip(lx1, 0, width)
    lx2 = np.clip(lx2, 0, width)
    rx1 = np.clip(rx1, 0, width)
    rx2 = np.clip(rx2, 0, width)
    
    # 8. 차선이 교차하거나 너무 가까워 유효하지 않은 경우 필터링:
    if lx1 >= rx1 or lx2 >= rx2:
        print("Warning: Detected lanes are crossing or too close for ROI creation. Skipping ROI.")
        return None

    # 9. 최종 사다리꼴 ROI 꼭짓점 반환:
    return np.array([[
        (lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)
    ]], dtype=np.int32)

# --- 9. ROI 높이 설정 (객체 감지 경고용) ---
danger_bottom, danger_top = 360, 300  # 위험 영역의 이미지 Y좌표 (하단, 상단).
warning_bottom, warning_top = 360, 260 # 경고 영역의 이미지 Y좌표 (하단, 상단).

# --- 10. ROI 별 객체 포함 임계값 설정 ---
danger_threshold = 0.2    # 객체 박스의 20% 이상이 위험 ROI에 포함되면 위험.
warning_threshold = 0.3   # 객체 박스의 30% 이상이 경고 ROI에 포함되면 경고.

# --- 11. 거리 계산 함수 ---
def calculate_distance(vehicle_screen_width, vehicle_real_length):
    # 핀홀 카메라 모델을 사용하여 객체까지의 거리를 추정합니다.
    if vehicle_screen_width == 0: # 0으로 나누는 오류 방지.
        return float('inf')
    return (focal_length * vehicle_real_length) / vehicle_screen_width

# --- 12. 트랙바 콜백 함수 ---
# 트랙바 위치에 따라 동영상 재생 위치를 업데이트합니다.
def update_video_position(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)

# --- 13. 메인 윈도우 생성 및 트랙바 추가 ---
cv2.namedWindow('YOLOv8 ROI Detection')
cv2.createTrackbar('Video Position', 'YOLOv8 ROI Detection', 0, video_length - 1, update_video_position)

# --- 14. FPS 측정을 위한 변수 초기화 ---
prev_frame_time = 0
new_frame_time = 0

# --- 15. 메인 비디오 처리 루프 ---
while cap.isOpened():
    ret, frame = cap.read() # 프레임 읽기.
    if not ret:
        break # 동영상 끝 또는 오류 발생 시 종료.

    frame = cv2.resize(frame, (resize_width, resize_height)) # 프레임 리사이즈.

    # --- 16. 객체 감지 필터링을 위한 ROI 마스크 생성 (현재 프레임 차선 기반) ---
    mask_danger = np.zeros((resize_height, resize_width), dtype=np.uint8)
    mask_warning = np.zeros((resize_height, resize_width), dtype=np.uint8)

    # danger ROI 생성 및 마스크 채우기.
    current_danger_roi = create_trapezoid_roi(frame, danger_bottom, danger_top)
    if current_danger_roi is not None:
        cv2.fillPoly(mask_danger, [current_danger_roi], 255)
        last_display_danger_roi = current_danger_roi # 유효한 ROI를 전역 변수에 저장.
        # --- 디버깅: Danger ROI 생성 성공 메시지 ---
        # print("Debug: Danger ROI created successfully.")
    # else:
        # print("Debug: Danger ROI creation failed for current frame.")

    # warning ROI 생성 및 마스크 채우기.
    current_warning_roi = create_trapezoid_roi(frame, warning_bottom, warning_top)
    if current_warning_roi is not None:
        cv2.fillPoly(mask_warning, [current_warning_roi], 255)
        # danger ROI와 겹치는 부분 제외 (마스크 빼기).
        if current_danger_roi is not None:
            danger_mask_for_subtract = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(danger_mask_for_subtract, [current_danger_roi], 255)
            mask_warning = cv2.subtract(mask_warning, danger_mask_for_subtract)
        last_display_warning_roi = current_warning_roi # 유효한 ROI를 전역 변수에 저장.
        # --- 디버깅: Warning ROI 생성 성공 메시지 ---
        # print("Debug: Warning ROI created successfully.")
    # else:
        # print("Debug: Warning ROI creation failed for current frame.")

    # --- 17. ROI 시각화를 위한 오버레이 프레임 준비 ---
    roi_overlay = frame.copy()

    # --- 18. ROI 그리기 로직 (last_display_roi 사용) ---
    # 차선 인식이 일시적으로 실패해도 이전에 성공한 ROI를 계속 그려줍니다.
    if last_display_danger_roi is not None:
        cv2.polylines(roi_overlay, [last_display_danger_roi], isClosed=True, color=(0, 0, 255), thickness=3) # 빨간색.

    if last_display_warning_roi is not None:
        # Warning ROI 테두리를 그릴 때 Danger ROI와 겹치지 않게 처리.
        if last_display_danger_roi is not None:
            warning_mask_draw = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(warning_mask_draw, [last_display_warning_roi], 255)

            danger_mask_draw = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(danger_mask_draw, [last_display_danger_roi], 255)

            warning_mask_draw = cv2.subtract(warning_mask_draw, danger_mask_draw) # 겹치는 부분 제거.

            # 윤곽선을 찾아 그립니다 (빼기 연산 후에는 윤곽선으로 그려야 깔끔).
            contours, _ = cv2.findContours(warning_mask_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3) # 노란색.
        else:
            # Danger ROI가 없으면 Warning ROI 전체를 그립니다.
            cv2.polylines(roi_overlay, [last_display_warning_roi], isClosed=True, color=(0, 255, 255), thickness=3)

    # --- 19. 객체 탐지 (YOLOv8) ---
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() # 바운딩 박스 좌표.
    classes = results[0].boxes.cls.cpu().numpy() # 클래스 ID.

    for box, class_id in zip(boxes, classes):
            class_id = int(class_id)
            if class_id not in class_names:
                continue # 정의되지 않은 클래스 ID는 건너뛰기

            x1, y1, x2, y2 = [int(c) for c in box]
            class_name = class_names[class_id]

        # 객체가 ROI에 속하는지에 따라 색상 및 경고 메시지 결정.
        # `mask_danger`, `mask_warning`은 `current_danger_roi`/`current_warning_roi` 기반입니다.
            if inside_roi(box, mask_danger, danger_threshold):
                color = (0, 0, 255) # 빨강 (위험).
            # --- 디버깅: 객체가 위험 ROI에 진입 ---
            # print(f"Debug: Object '{class_name}' entered DANGER ROI.")
            
            # 위험 영역에 있을 때만 박스와 텍스트를 그림
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 차량의 화면 상 너비 계산 (위험/경고 시에만 거리 표시)
                vehicle_screen_width = x2 - x1
                if class_id == 0 or class_id == 1: # "vehicle" 또는 "big vehicle"
                    vehicle_real_length_used = vehicle_real_length if class_id == 0 else big_vehicle_real_length
                    if vehicle_screen_width > 0:
                        distance = calculate_distance(vehicle_screen_width, vehicle_real_length_used)
                    else:
                        distance = float('inf') # 화면 너비가 0이면 무한대 거리로 설정
                    cv2.putText(frame, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                elif inside_roi(box, mask_warning, warning_threshold):
                    color = (0, 255, 255) # 노랑 (경고).
            # --- 디버깅: 객체가 경고 ROI에 진입 ---
            # print(f"Debug: Object '{class_name}' entered WARNING ROI.")
            
            # 경고 영역에 있을 때만 박스와 텍스트를 그림
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 차량의 화면 상 너비 계산 (위험/경고 시에만 거리 표시)
                vehicle_screen_width = x2 - x1
                if class_id == 0 or class_id == 1: # "vehicle" 또는 "big vehicle"
                    vehicle_real_length_used = vehicle_real_length if class_id == 0 else big_vehicle_real_length
                    if vehicle_screen_width > 0:
                        distance = calculate_distance(vehicle_screen_width, vehicle_real_length_used)
                    else:
                        distance = float('inf') # 화면 너비가 0이면 무한대 거리로 설정
                    cv2.putText(frame, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            else:
            # 객체가 위험 또는 경고 ROI에 속하지 않을 경우 아무것도 그리지 않음
                pass # 이 부분에 아무 코드도 넣지 않으면 됨
            # 기존 초록색 바운딩 박스 그리는 코드를 제거합니다:
            # color = (0, 255, 0) # 초록 (정상).
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, class_name, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 거리 계산 로직도 제거하거나 위 조건부 블록 안으로 옮깁니다.
            # 이 else 블록은 실행되지 않으므로, 객체가 ROI에 포함되지 않으면 아래 거리 계산도 실행되지 않습니다.
            # if class_id == 0 or class_id == 1: # "vehicle" 또는 "big vehicle".
            #     vehicle_real_length_used = vehicle_real_length if class_id == 0 else big_vehicle_real_length
            #     if vehicle_screen_width > 0:
            #         distance = calculate_distance(vehicle_screen_width, vehicle_real_length_used)
            #     else:
            #         distance = float('inf') # 화면 너비가 0이면 무한대 거리로 설정.
            #     cv2.putText(frame, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    # --- 20. FPS 계산 및 화면 표시 ---
    new_frame_time = time.time()
    fps_value = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 21. 최종 화면 표시 (원본 프레임과 ROI 오버레이 합치기) ---
    overlay = cv2.addWeighted(frame, 1.0, roi_overlay, 0.3, 0) # 원본 프레임에 ROI를 투명하게 겹쳐 그림.
    cv2.imshow("YOLOv8 ROI Detection", overlay)

    # --- 22. 사용자 입력 처리 ---
    key = cv2.waitKey(delay) & 0xFF # 'delay' 밀리초 동안 대기.
    if key == ord('q'): # 'q' 키를 누르면 종료.
        break
    elif key == ord('Q'): # 'Q' 키를 누르면 5초 뒤로 이동.
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 5000)
    elif key == ord('S'): # 'S' 키를 누르면 5초 앞으로 이동.
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 5000)

# --- 23. 자원 해제 ---
cap.release() # 비디오 캡처 객체 해제.
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기.