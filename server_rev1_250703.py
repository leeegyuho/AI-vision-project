import socket
import struct
import cv2
import numpy as np
import time
import json
from ultralytics import YOLO
import traceback

# --- 설정 ---
FOCAL_LENGTH = 600

REAL_HEIGHTS = {
    "person": 1.6, "car": 1.5, "bus": 3.2, "truck": 3.4,
    "motorbike": 1.4, "bicycle": 1.2, "vehicle": 1.5,
    "big vehicle": 3.5, "bike": 1.2, "human": 1.7,
    "animal": 0.5, "obstacle": 1.0
}

REAL_WIDTHS = {
    "person": 0.5, "car": 1.8, "bus": 2.5, "truck": 2.5,
    "motorbike": 0.8, "bicycle": 0.7, "vehicle": 1.8,
    "big vehicle": 2.5, "bike": 0.5, "human": 0.5,
    "animal": 0.6, "obstacle": 1.0
}

def estimate_distance(h, w, label):
    try:
        dist_h = (REAL_HEIGHTS[label] * FOCAL_LENGTH) / h
        dist_w = (REAL_WIDTHS[label] * FOCAL_LENGTH) / w
        return (dist_h + dist_w) / 2
    except:
        return -1

# YOLO 모델 로드
model = YOLO("/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt")
#model.to('cuda')

# ROI 및 FPS 관련 전역 변수 초기화
prev_frame_time = 0
prev_edges = None
frame_count = 0
roi_update_interval = 5  # ROI 업데이트 간격 (프레임 단위)
prev_danger_roi = None
prev_warning_roi = None

# ROI 마스크 포함 임계값 설정 (이 값들은 그대로 유지)
danger_threshold = 0.1
warning_threshold = 0.1

# 바운딩 박스가 ROI 마스크 내 일정 비율 이상 포함되었는지 확인
def inside_roi(box, mask, threshold):
    x1, y1, x2, y2 = map(int, box)
    roi_box = mask[y1:y2, x1:x2]
    if roi_box.size == 0:
        return False
    inside = np.count_nonzero(roi_box == 255)
    return inside / roi_box.size >= threshold

# 그림자 제거용 HSV 채널 필터링 함수
def remove_shadows_color_based(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([179, 19, 68])
    shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
    result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(shadow_mask))
    return result

# 차선 기반 trapezoid 형태의 ROI 생성 함수
def create_trapezoid_roi(frame, y_bottom, y_top):
    global prev_edges
    height, width = frame.shape[:2]
    # 그레이스케일 변환 및 에지 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 140)
    # 이전 에지 맵과 가중 평균하여 안정화
    if prev_edges is not None:
        edges = cv2.addWeighted(edges.astype(np.float32), 0.7,
                                prev_edges.astype(np.float32), 0.3, 0).astype(np.uint8)
    prev_edges = edges.copy()
    # 관심영역 설정 (YOLO 감지를 위한 차선 마스크)
    mask = np.zeros_like(edges)
    # **이 roi_vertices의 비율을 필요에 따라 추가 조정 가능합니다.**
    # (width * 0.05, height) 예시는 더 넓게 설정한 것이며, 시각적 확인 후 미세 조정 필요
    roi_vertices = np.array([[
        (int(width * 0.1), height),          # 좌측 하단 (이미지 바닥)
        (int(width * 0.45), int(height * 0.5)), # 좌측 상단 (도로 안쪽) - 여기 0.5로 이미 조정됨
        (int(width * 0.55), int(height * 0.5)), # 우측 상단 (도로 안쪽) - 여기 0.5로 이미 조정됨
        (int(width * 0.9), height)           # 우측 하단 (이미지 바닥)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    # ROI 내 차선 검출
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=70)
    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5: left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5: right_lines.append((x1, y1, x2, y2))
    # 직선 근사로 평균화
    def average_line(lines):
        if not lines: return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]; y += [y1, y2]
        # x, y 데이터가 충분해야 polyfit이 작동
        if len(x) < 2: return None # 최소 2개의 점 필요
        return np.polyfit(y, x, 1) # y를 독립변수, x를 종속변수로 (세로선에 유리)

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)

    # 차선이 감지되지 않으면 None 반환
    if left_fit is None or right_fit is None:
        # 이 경우, 이전 프레임의 ROI를 재사용하도록 prev_danger_roi/prev_warning_roi를 활용합니다.
        return None

    # 감지된 차선을 바탕으로 ROI 폴리곤 정의
    # Y-bottom과 Y-top은 호출 시 전달된 값 (동적으로 계산된 값) 사용
    lx1, lx2 = int(np.polyval(left_fit, y_bottom)), int(np.polyval(left_fit, y_top))
    rx1, rx2 = int(np.polyval(right_fit, y_bottom)), int(np.polyval(right_fit, y_top))

    # 차선 ROI가 화면 밖으로 나가지 않도록 클램핑
    lx1 = np.clip(lx1, 0, width)
    lx2 = np.clip(lx2, 0, width)
    rx1 = np.clip(rx1, 0, width)
    rx2 = np.clip(rx2, 0, width)

    # 유효한 좌표인지 확인
    if not (0 <= lx1 <= width and 0 <= lx2 <= width and \
            0 <= rx1 <= width and 0 <= rx2 <= width):
        return None # 유효하지 않은 ROI는 반환하지 않음

    return np.array([[(lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)]], dtype=np.int32)


# 소켓 설정
HOST = '0.0.0.0'
PORT = 7777

def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise ConnectionError("클라이언트와 연결이 끊어졌습니다.")
        data += more
    return data

# 소켓 수신 대기
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"INFO: 서버가 {HOST}:{PORT}에서 대기 중입니다...")
conn, addr = server_socket.accept()
print(f"INFO: 클라이언트 연결됨: {addr}")

prev_time = time.time()

# `danger_bottom`, `danger_top` 등 Y좌표 값들을 루프 밖에서 초기화할 필요 없음
# 루프 안에서 동적으로 계산됩니다.

# 출력 창 크기 지정 (추가)
cv2.namedWindow("YOLO + Polygon Danger Detection", cv2.WINDOW_NORMAL)
# 클라이언트가 보내는 해상도에 맞춰 크기 설정 (예: 640x480)
# 서버 화면의 초기 크기를 640x480으로 고정하여 일관성 유지
cv2.resizeWindow("YOLO + Polygon Danger Detection", 640, 480)


while True:
    try:
        # 프레임 수신
        length_buf = recvall(conn, 4)
        if not length_buf:
            break
        frame_len = struct.unpack('>I', length_buf)[0]
        frame_data = recvall(conn, frame_len)
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        danger_objects = []
        height, width = frame.shape[:2] # 현재 프레임의 높이와 너비를 가져옴

        # --- 변경 사항: danger_bottom, danger_top 등을 프레임 높이에 맞춰 동적으로 계산 ---
        # 이 비율(0.95, 0.55 등)을 원하는 ROI 위치에 맞춰 미세 조정하세요.
        # 예시 값들은 ROI를 더 넓고 위쪽까지 포함하도록 조정한 것입니다.
        dynamic_danger_bottom = int(height * 1) # 프레임 높이의 95% 지점 (하단에 가깝게)
        dynamic_danger_top = int(height * 0.8)   # 프레임 높이의 55% 지점 (중앙보다 약간 아래)

        dynamic_warning_bottom = int(height * 0.8) # 프레임 높이의 99% 지점 (거의 바닥)
        dynamic_warning_top = int(height * 0.6)    # 프레임 높이의 45% 지점 (중앙보다 약간 위)
        # ----------------------------------------------------------------------


        # ROI 갱신 (간격마다)
        if frame_count % roi_update_interval == 0:
            # create_trapezoid_roi 호출 시 위에서 계산한 동적 Y좌표 사용
            danger_roi = create_trapezoid_roi(frame, dynamic_danger_bottom, dynamic_danger_top)
            warning_roi = create_trapezoid_roi(frame, dynamic_warning_bottom, dynamic_warning_top)
            if danger_roi is not None:
                prev_danger_roi = danger_roi
            if warning_roi is not None:
                prev_warning_roi = warning_roi
        else:
            danger_roi = prev_danger_roi
            warning_roi = prev_warning_roi
        
        # ROI 마스크 생성
        mask_danger = np.zeros((height, width), dtype=np.uint8)
        mask_warning = np.zeros((height, width), dtype=np.uint8)
        
        # 유효한 ROI가 있을 경우에만 마스크 채우기
        if danger_roi is not None:
            cv2.fillPoly(mask_danger, [danger_roi], 255)
        if warning_roi is not None:
            cv2.fillPoly(mask_warning, [warning_roi], 255)
            mask_warning = cv2.subtract(mask_warning, mask_danger)  # 위험 ROI 제외


        # 시각화용 프레임 복사
        roi_overlay = frame.copy() # ROI를 그릴 투명 오버레이
        frame_output = frame.copy() # YOLO 결과 및 FPS 등을 그릴 프레임

        # ROI 시각화 (빨강/노랑 폴리라인)
        if danger_roi is not None:
            cv2.polylines(roi_overlay, [danger_roi], isClosed=True, color=(0, 0, 255), thickness=3)
        if warning_roi is not None:
            # warning_mask는 이미 위에서 계산되었지만, 시각화를 위해 다시 사용
            # danger_mask도 필요하므로 다시 정의
            warning_mask_for_viz = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(warning_mask_for_viz, [warning_roi], 255)
            danger_mask_for_viz = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(danger_mask_for_viz, [danger_roi], 255)
            warning_mask_for_viz = cv2.subtract(warning_mask_for_viz, danger_mask_for_viz)
            
            contours, _ = cv2.findContours(warning_mask_for_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3)
        
        # 객체 감지 수행
        results = model(frame, conf=0.3)[0]
        boxes = results.boxes
        
        # 객체 정보 추출 및 시각화 (YOLO 감지 결과)
        if boxes is not None and boxes.xyxy is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                label_id = int(boxes.cls[i].item())
                label = model.names[label_id]

                if label not in REAL_HEIGHTS:
                    continue

                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                cx, cy = (x1 + x2) // 2, y2
                distance = estimate_distance(h, w, label)

                in_danger = inside_roi((x1, y1, x2, y2), mask_danger, danger_threshold)
                in_warning = inside_roi((x1, y1, x2, y2), mask_warning, warning_threshold)

                # --- 서버 화면 디버깅용 시각화 (모든 객체 표시) ---
                # 객체 바운딩 박스 및 텍스트 색상 결정 (초록색 추가)
                display_color_bgr = (0, 0, 255) if in_danger else \
                                    (0, 255, 255) if in_warning else \
                                    (0, 255, 0)

                box_thickness = 1
                label_font_scale = 0.7
                label_font_thickness = 1
                distance_font_scale = 0.5
                distance_font_thickness = 1
                shadow_color = (0, 0, 0)
                shadow_offset = 1

                # 바운딩 박스 그리기
                cv2.rectangle(frame_output, (x1, y1), (x2, y2), display_color_bgr, box_thickness)

                # 객체 이름 (레이블) 표시 (그림자 효과)
                label_text_pos = (x1, y1 - 10)
                cv2.putText(frame_output, label,
                            (label_text_pos[0] + shadow_offset, label_text_pos[1] + shadow_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, shadow_color, label_font_thickness + 1)
                cv2.putText(frame_output, label, label_text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, display_color_bgr, label_font_thickness)

                # 객체 중앙점 (점) 표시
                cv2.circle(frame_output, (cx, cy), 5, display_color_bgr, -1)

                # 거리 표시 (그림자 효과)
                distance_text_pos = (x1, y2 + 20)
                cv2.putText(frame_output, f"{distance:.2f}m",
                            (distance_text_pos[0] + shadow_offset, distance_text_pos[1] + shadow_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, distance_font_scale, shadow_color, distance_font_thickness + 1)
                cv2.putText(frame_output, f"{distance:.2f}m", distance_text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, distance_font_scale, display_color_bgr, distance_font_thickness)


                # --- 클라이언트에게 전송할 데이터 구성 (ROI 내 객체만 필터링) ---
                if in_danger or in_warning:
                    danger_objects.append({
                        "label": str(label),
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(w),
                        "h": int(h),
                        "distance": float(round(distance, 2)),
                        "zone": "red" if in_danger else "yellow"
                    })

        # FPS 측정 및 표시
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        text_pos = (10, 40)
        font_scale = 0.8
        font_thickness = 2
        font_color = (255, 255, 0) # 하늘색 (BGR)
        shadow_color = (0, 0, 0)   # 검은색 그림자
        shadow_offset = 2          # 그림자 오프셋 (픽셀)

        # 그림자 텍스트 그리기
        cv2.putText(frame_output, f"FPS: {fps:.2f}", (text_pos[0] + shadow_offset, text_pos[1] + shadow_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, font_thickness)
        # 실제 텍스트 그리기
        cv2.putText(frame_output, f"FPS: {fps:.2f}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

        # 클라이언트에게 전송 (JSON 직렬화)
        try:
            response_json = json.dumps(danger_objects)
        except Exception as e:
            print(f"[JSON 직렬화 에러] {e}")
            print("문제가 된 데이터:", danger_objects)
            continue
        
        payload = response_json.encode('utf-8')
        conn.sendall(struct.pack('>I', len(payload)))  # 4바이트 길이
        conn.sendall(payload)

        # 디버깅 화면 출력 (오버레이 합성 후)
        final_display = cv2.addWeighted(frame_output, 1.0, roi_overlay, 0.3, 0)
        cv2.imshow("YOLO + Polygon Danger Detection", final_display)
        
        # 프레임 카운트 증가 (ROI 업데이트 간격 계산용)
        frame_count += 1 

        if cv2.waitKey(1) == 27: # ESC 키로 종료
            break

    except ConnectionError as ce:
        print(f"[연결 끊김 에러] {ce}")
        break # 연결 끊김 시 루프 종료
    except Exception as e:
        print(f"[일반 에러] {e}")
        traceback.print_exc()
        break # 다른 예상치 못한 에러 발생 시 루프 종료

# 자원 해제
conn.close()
server_socket.close()
cv2.destroyAllWindows()