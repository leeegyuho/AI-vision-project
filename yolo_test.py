from ultralytics import YOLO
import cv2

SCALE = 0.4 # 프레임 축소 비율 (0.5 = 50%)
    
# 모델 로드
model = YOLO("/home/hkit/Downloads/yolov8_custom14/weights/best.pt")  # 학습된 모델 경로

# 비디오 로드
video_path = "/home/hkit/Downloads/test_video/rural_cut.webm"  # 테스트 영상
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 축소
    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

    # YOLO로 객체 탐지
    results = model(frame)[0]  # results는 Results 객체 리스트, [0]으로 첫 결과 선택

    # 결과에서 박스와 클래스 정보 가져오기
    for box in results.boxes:
        cls_id = int(box.cls[0])  # 클래스 인덱스
        conf = float(box.conf[0])  # confidence
        label = model.names[cls_id]  # 클래스 이름 (자동차, 사람 등)

        # 바운딩 박스 좌표
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌상단(x1, y1), 우하단(x2, y2)

        # 박스와 라벨 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 영상 출력
    cv2.imshow("YOLOv8 Detection", frame)

    # ESC키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()