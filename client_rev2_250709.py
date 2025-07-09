# FPS 클라이언트 - 서버 간 딜레이 개선
# FPS 속도 개선 - JPEG_QUALITY값 조절
# UI 추가 - warning(마름모), danger(삼각형)

import cv2
import socket
import struct
import numpy as np
import time
import json  # JSON 파싱을 위한 모듈

# --- 설정 ---
#SERVER_IP = '192.168.3.28' #내 노트북 서버 주소
SERVER_IP = '127.0.0.1' #내 노트북 서버 주소
SERVER_PORT = 7777
#VIDEO_SOURCE = 'rural_cut.webm'
VIDEO_SOURCE = '/home/hkit/Pictures/video/rural_cut.webm'

resize_width, resize_height = 640, 480 #300

def main():
    fps = 0.0
    prev_frame_time = time.time()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"INFO: 서버({SERVER_IP}:{SERVER_PORT})에 성공적으로 연결되었습니다.")
    except socket.error as e:
        print(f"ERROR: 서버 연결에 실패했습니다: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: 비디오 소스를 열 수 없습니다: {VIDEO_SOURCE}")
        client_socket.close()
        return

    print("INFO: 클라이언트를 시작합니다. 'q' 키를 누르면 종료됩니다.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("INFO: 비디오 스트림의 끝에 도달했거나 오류가 발생했습니다.")
            break

        # 전송 전에 영상 축소 (640, 480)
        frame = cv2.resize(frame, (resize_width, resize_height))

        # JPEG 인코딩 품질 (60->40) 낮추어 FPS 향상
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            print("WARNING: 프레임 인코딩에 실패했습니다.")
            continue

        data = encoded_frame.tobytes()

        try:
            client_socket.sendall(struct.pack('>I', len(data)))
            client_socket.sendall(data)

            len_buf = client_socket.recv(4, socket.MSG_WAITALL)
            if not len_buf:
                print("WARNING: 서버로부터 응답 길이를 받지 못했습니다.")
                continue
            response_len = struct.unpack('>I', len_buf)[0]

            response = client_socket.recv(response_len, socket.MSG_WAITALL).decode('utf-8')

            try:
                objects = json.loads(response)

                # --- 폰트 및 그림자 설정 ---
                label_font_scale = 0.7
                label_font_thickness = 1

                distance_font_scale = 0.5
                distance_font_thickness = 1

                shadow_color = (0, 0, 0)   # 검은색 그림자
                shadow_offset = 1          # 그림자 오프셋 (픽셀)

                for obj in objects:
                    label = obj.get("label", "unknown")
                    x = obj.get("x", 0)
                    y = obj.get("y", 0)
                    w = obj.get("w", 0)
                    h = obj.get("h", 0)
                    dist = obj.get("distance", -1)
                    zone = obj.get("zone", "red")

                    color = (0, 0, 255) if zone == "red" else (0, 255, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    label_text_pos = (x, y - 10)
                    cv2.putText(frame, label,
                                (label_text_pos[0] + shadow_offset, label_text_pos[1] + shadow_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, shadow_color, label_font_thickness + 1)
                    cv2.putText(frame, label, label_text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, color, label_font_thickness)

                    distance_text_pos = (x, y + h + 20)
                    cv2.putText(frame, f"Dis: {dist:.2f}m",
                                (distance_text_pos[0] + shadow_offset, distance_text_pos[1] + shadow_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, distance_font_scale, shadow_color, distance_font_thickness + 1)
                    cv2.putText(frame, f"Dis: {dist:.2f}m", distance_text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, distance_font_scale, color, distance_font_thickness)

                # ===================== ✅ UI 네모 박스 표시용 코드 =====================
                # 이 코드는 frame 위에 좌측 상단에 객체 정보를 네모 박스 안에 표시합니다.

                # 표시할 텍스트 준비 (objects는 서버로부터 받은 객체 리스트)
                info_lines = []
                for obj in objects:
                    label = obj.get("label", "unknown")
                    dist = obj.get("distance", -1)
                    zone = obj.get("zone", "red")
                    zone_label = "danger" if zone == "red" else "warning" if zone == "yellow" else zone
                    info_lines.append(f"{label} {dist:.2f}m [{zone_label}]")

                # 🔻 FPS 정보도 추가 (맨 위에 출력되게 함)
                info_lines.insert(0, f"FPS: {fps:.2f}")

                # 텍스트 박스 위치 및 스타일 설정
                box_x, box_y = 0, 0               # 좌측 상단 위치
                line_height = 75                  # 한 줄 높이
                padding = 0                       # 위쪽 간격
                box_width = 150                   # 너비 고정 또는 max 길이 기준으로 설정 가능
                # box_height = padding * 2 + line_height * len(info_lines)

                # # 배경 박스 그리기 (반투명 또는 불투명)
                # cv2.rectangle(frame, (box_x, box_y),
                #               (box_x + box_width, box_y + box_height),
                #               (50, 50, 50), thickness=-1)  # 채운 사각형 (어두운 배경)

                # cv2.rectangle(frame, (box_x, box_y),
                #               (box_x + box_width, box_y + box_height),
                #               (200, 200, 200), thickness=1)  # 외곽 테두리 (회색)

                # 텍스트 쓰기 (FPS는 흰색, 객체는 zone에 따라 색상 다르게)
                # for i, line in enumerate(info_lines):
                #     y = box_y + padding + i * line_height + 15

                #     if i == 0:
                #         color = (255, 255, 255)  # FPS는 흰색
                #     else:
                #         zone = objects[i - 1].get("zone", "red")
                #         if zone == "red":
                #             color = (0, 0, 255)
                #         elif zone == "yellow":
                #             color = (0, 255, 255)
                #         else:
                #             color = (0, 255, 0)

                #     cv2.putText(frame, line, (box_x + 8, y),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                for i, line in enumerate(info_lines):
                    # 각 줄의 텍스트 크기 및 높이 계산
                    text_size, baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    text_width, text_height = text_size

                    # 좌측 상단 박스에서 y 위치 계산 (중앙 정렬을 위해)
                    y_top = box_y + padding + i * line_height
                    y_center = y_top + line_height // 2
                    x_center = box_x + box_width // 2

                    # 텍스트를 중앙에 위치시키기 위한 계산 (수평 정렬)
                    text_x = x_center - text_width // 2
                    text_y = y_center + text_height // 2 - baseline // 2  # baseline을 고려한 수직 중앙 정렬

                    # FPS 라인 출력 (흰색 텍스트만 출력)
                    if i == 0:
                        cv2.putText(frame, line, (box_x + 10, y_center + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                        continue

                    # 텍스트를 다룰 객체 정보 가져오기
                    obj = objects[i - 1]
                    zone = obj.get("zone", "red")

                    if zone == "red":
                        # 🔺 빨간색 삼각형
                        triangle_pts = np.array([
                            [x_center, y_top + 10],                            # 위쪽 꼭짓점
                            [box_x + 10, y_top + line_height - 10],           # 왼쪽 아래
                            [box_x + box_width - 10, y_top + line_height - 10]# 오른쪽 아래
                        ], np.int32)
                        cv2.fillPoly(frame, [triangle_pts], color=(0, 0, 255))

                        # 도형 위에 'danger' 텍스트
                        danger_text = "DANGER"
                        dz_size, _ = cv2.getTextSize(danger_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dz_x = x_center - dz_size[0] // 2
                        dz_y = triangle_pts[0][1] - 8
                        cv2.putText(frame, danger_text, (dz_x, dz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        # 도형 내부에 객체 정보 텍스트
                        content_text = f"{obj.get('label', 'unknown')} {obj.get('distance', -1):.2f}m"
                        ct_size, _ = cv2.getTextSize(content_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        ct_x = x_center - ct_size[0] // 2
                        # ct_y = y_top + line_height // 2 + ct_size[1] // 2  # 삼각형 중앙 배열
                        ct_y = y_top + line_height - 10 # 바닥쪽으로 붙이기 (-10 : 살짝 띄우기)
                        cv2.putText(frame, content_text, (ct_x, ct_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    elif zone == "yellow":
                        # 🔶 노란색 마름모
                        y_center = y_top + line_height // 2
                        diamond_pts = np.array([
                            [x_center, y_top + 10],                           # 위
                            [box_x + box_width - 10, y_center],              # 오른쪽
                            [x_center, y_top + line_height - 10],            # 아래
                            [box_x + 10, y_center]                           # 왼쪽
                        ], np.int32)
                        cv2.fillPoly(frame, [diamond_pts], color=(0, 255, 255))

                        # 도형 위에 'warning' 텍스트
                        warn_text = "WARNING"
                        wz_size, _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        wz_x = x_center - wz_size[0] // 2
                        wz_y = diamond_pts[0][1] - 8
                        cv2.putText(frame, warn_text, (wz_x, wz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        # 도형 내부에 객체 정보 텍스트
                        content_text = f"{obj.get('label', 'unknown')} {obj.get('distance', -1):.2f}m"
                        ct_size, _ = cv2.getTextSize(content_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        ct_x = x_center - ct_size[0] // 2
                        ct_y = y_top + line_height // 2 + ct_size[1] // 2
                        cv2.putText(frame, content_text, (ct_x, ct_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


                    else:
                        # 🟩 기타 - 초록색 텍스트만
                        cv2.putText(frame, line, (text_x, text_y),  # 텍스트 중앙에 위치
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

                # =====================================================================

            except json.JSONDecodeError as e:
                print(f"WARNING: JSON 디코딩 오류 발생: {e}, 수신 데이터: '{response}'")

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_frame_time)
            prev_frame_time = curr_time

        except socket.error as e:
            print(f"ERROR: 소켓 통신 오류: {e}")
            break

        cv2.imshow('Client View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("INFO: 자원을 해제하고 클라이언트를 종료합니다.")
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
