# Line detection & Collision detection (추돌 감지)

File 
1. YOLOv8_custiom14_8.zip
    YOLOv8 model results임
     metrics/precision : 0.802
     metrics/recall : 0.701  
     metrics/mAP50  : 0.767  
     metrics/mAP50-95: 0.483

2. collsion_warning
   차선 인식 없이 roi영역 설정 후 추돌 방지 알림

3. collsion_warning_rev2.py
   차선 인식에 따른 roi영역 설정 후 추돌 감지 

4. yolo_test
  모델 테스트 코드

5. client_rev1_250703.py, server_rev1_250703.py
   python TCP/IP 통신-------FPS 낮음
6. CC++_CONNECTING_rev2.cpp
   C/C++ 통신 하였고 
   FP32로하였음-- convert file yolo.py를 c언어로 바꾸는 것 
