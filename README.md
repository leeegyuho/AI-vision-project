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
8. CC++_CONNECTING_rev2.cpp
   C/C++ 통신 하였고 
   FP32로하였음-- convert file yolo.py를 c언어로 바꾸는 것 

YOLOv8 커스텀 모델 선정

CLASS
vehicle, big vehicle, bike, human, animal, obstacle

DATE 
Train : test : val = 7 : 2 : 1
(620ea : 196ea : 124ea)

Result

    metrics/precision : 0.802
    metrics/recall : 0.701  
    metrics/mAP50  : 0.767  
    metrics/mAP50-95: 0.483

문제점    
데이터 중복으로 인한 과적합
사람인식 오류
bike, human 객체 인식 혼동
가깝거나 멀어지면 인식 안됌
빌딩을 bike나 vehicle로 인식

해결방안
한 사진에서 여러 클래스 라벨링(중복데이터 제거)
저화질이거나 애매한 사진 제거 
양질의 데이터 추가 (각 클래스)
bike의 경우 bike만 라벨링
클래스 간소화(8개 -> 6개)


차선 인식 및 추돌 감지

OpenCV로 영상처리
