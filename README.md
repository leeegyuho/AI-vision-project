# Thallos project - 차선인식에 따른 전방 추돌 감지 

File 
1. YOLOv8_custiom14_8.zip - 커스텀 모델 선정 

2. collsion_warning
   차선 인식 없이 roi영역 설정 후 추돌 방지 알림

3. collsion_warning_rev2.py
   차선 인식에 따른 roi영역 설정 후 추돌 감지

4. final_collision_detection.py
   차선인식에 따른 ROI영역 설정 후 추돌 감지

6. yolo_test
  모델 테스트 코드

7. client_rev2_250709.py, server_rev1_250709.py
   python TCP/IP 통신-------FPS 낮음
   경고, 위험 알림 추가 디자인

8.thallos_ros 
   ROS2

＊YOLOv8 커스텀 모델 선정＊

1. CLASS
   
vehicle, big vehicle, bike, human, animal, obstacle

3. DATE
   
   Train : test : val = 7 : 2 : 1
   (620ea : 196ea : 124ea)

   Result

    metrics/precision : 0.802
    metrics/recall : 0.701  
    metrics/mAP50  : 0.767  
    metrics/mAP50-95: 0.483

5. 문제점
   
   데이터 중복으로 인한 과적합
   사람인식 오류
   bike, human 객체 인식 혼동
   가깝거나 멀어지면 인식 안됌
   빌딩을 bike나 vehicle로 인식

7. 해결방안
   
   한 사진에서 여러 클래스 라벨링(중복데이터 제거)
   저화질이거나 애매한 사진 제거 
   양질의 데이터 추가 (각 클래스)
   bike의 경우 bike만 라벨링
   클래스 간소화(8개 -> 6개)


＊차선 인식 및 추돌 감지＊

OpenCV로 영상처리

1.주요 기능

차선 인식
:캐니 엣지, 히스토그램, HSV, 가중치(7:3),ROI, HoughlinesP, Average_line, Slope(기울기에 따라 왼쪽, 오른쪽 차선)

추돌 감지
: 차선인식에 따른 WARNING, DANGER 영역 설정 

차간 거리
: 초점거리 600, 실제 객체 높이, 너비는 임의로 지정 -> 높이, 너비에 따라서 차간 거리 더함 / 2

2. 통신 (Python)
   
   카메라-96보드-데스크탑
   TCP/IP
   카메라에서 96보드와 USB연결로 영상받는 걸로 가정
   
   멀티스레드 : One 영상 받음, The other 영상 데스크탑으로 전송

   JPEG로 인코딩 후 데스크탑에서 디코딩하고 WARNING, DANGER에서 감지되는 객체만 JSON으로 데이터 전송



<img width="1326" height="656" alt="image" src="https://github.com/user-attachments/assets/06209951-debd-4544-b4bc-984d8eaa53e8" />

<img width="1287" height="648" alt="image" src="https://github.com/user-attachments/assets/c480fbd3-ced2-4398-a0a7-c5c045cf93a7" />

<img width="1346" height="535" alt="image" src="https://github.com/user-attachments/assets/5aae8472-d194-433e-84d2-3bf1182e03ef" />


