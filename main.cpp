#include "server_socket.hpp"
#include "image_processor.hpp"
#include "yolo_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <nlohmann/json.hpp> // JSON 처리 라이브러리
#include <iomanip> // For std::fixed, std::setprecision

// --- 설정 (Python 코드의 상수와 일치) ---
// YOLO 모델 경로 (여러분의 OpenVINO IR 모델 경로로 변경해주세요)
const std::string MODEL_XML_PATH = "/home/hkit/cc++/server/models/best.xml";
const std::string MODEL_BIN_PATH = "/home/hkit/cc++/server/models/best.bin";
const std::string DEVICE = "CPU"; // "CPU", "GPU" 등 OpenVINO 지원 디바이스

// ROI 및 감지 임계값
const float CONF_THRESHOLD = 0.3f;
const float NMS_THRESHOLD = 0.5f;
const float DANGER_THRESHOLD = 0.1f; // 바운딩 박스가 ROI 내 일정 비율 이상 포함되었는지 확인
const float WARNING_THRESHOLD = 0.1f;

// ROI 업데이트 간격 (프레임 단위)
const int ROI_UPDATE_INTERVAL = 5;

// 서버 IP 및 포트
const std::string HOST = "0.0.0.0"; // 모든 인터페이스에서 수신
const int PORT = 7777;

// --- 메인 함수 ---
int main() {
    // FPS 계산 변수
    double fps = 0.0;
    long frame_count = 0;
    auto prev_time = std::chrono::high_resolution_clock::now();

    // ROI 관련 변수
    cv::Mat prev_danger_roi_mask; // 이전 프레임의 Danger ROI 마스크 (재사용용)
    cv::Mat prev_warning_roi_mask; // 이전 프레임의 Warning ROI 마스크 (재사용용)
    int roi_frame_counter = 0; // ROI 업데이트 주기 카운터

    // 소켓 서버 초기화
    ServerSocket server_socket;
    if (!server_socket.start(HOST, PORT)) {
        return 1;
    }

    // YOLO Detector 초기화
    YoloDetector detector(MODEL_XML_PATH, MODEL_BIN_PATH, DEVICE, CONF_THRESHOLD, NMS_THRESHOLD);
    if (!detector.isInitialized()) {
        std::cerr << "ERROR: YOLO Detector 초기화 실패." << std::endl;
        server_socket.stop();
        return 1;
    }

    // 이미지 프로세서 초기화 (ROI 관련 함수 포함)
    ImageProcessor image_processor;

    // 출력 창 크기 지정 (Python 서버와 동일하게)
    cv::namedWindow("YOLO + Polygon Danger Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO + Polygon Danger Detection", 640, 480); // 클라이언트 해상도 480x360인데, 서버 디스플레이는 좀 더 크게 볼 수도 있음

    try {
        std::cout << "INFO: 서버가 클라이언트 연결을 기다립니다..." << std::endl;
        int client_sock_fd = server_socket.acceptClient(); // 클라이언트 연결 대기 및 수락
        if (client_sock_fd == -1) {
            std::cerr << "ERROR: 클라이언트 연결 수락 실패." << std::endl;
            server_socket.stop();
            return 1;
        }
        std::cout << "INFO: 클라이언트 연결됨: " << server_socket.getClientAddr() << std::endl;

        while (true) {
            // 1. 프레임 수신
            std::vector<uchar> encoded_frame_data;
            try {
                encoded_frame_data = server_socket.receiveFrame(client_sock_fd);
            } catch (const std::runtime_error& e) {
                std::cerr << "ERROR: 프레임 수신 중 오류: " << e.what() << std::endl;
                break; // 연결 끊김으로 간주하고 루프 종료
            }

            cv::Mat frame = cv::imdecode(encoded_frame_data, cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "WARNING: 수신된 프레임이 비어있거나 디코딩 실패." << std::endl;
                continue;
            }

            int height = frame.rows;
            int width = frame.cols;

            // --- 동적 ROI Y 좌표 계산 (Python 코드와 동일) ---
            int dynamic_danger_bottom = static_cast<int>(height * 1.0); // 프레임 높이의 100% 지점 (하단)
            int dynamic_danger_top = static_cast<int>(height * 0.9);    // 프레임 높이의 90% 지점 (하단보다 위)

            int dynamic_warning_bottom = static_cast<int>(height * 0.9); // 프레임 높이의 90% 지점
            int dynamic_warning_top = static_cast<int>(height * 0.7);    // 프레임 높이의 70% 지점
            // --------------------------------------------------

            // 2. ROI 갱신 (간격마다)
            cv::Mat current_danger_roi_mask, current_warning_roi_mask;
            if (roi_frame_counter % ROI_UPDATE_INTERVAL == 0) {
                std::vector<cv::Point> danger_roi_polygon = image_processor.createTrapezoidROI(frame, dynamic_danger_bottom, dynamic_danger_top);
                std::vector<cv::Point> warning_roi_polygon = image_processor.createTrapezoidROI(frame, dynamic_warning_bottom, dynamic_warning_top);

                if (!danger_roi_polygon.empty()) {
                    current_danger_roi_mask = cv::Mat::zeros(height, width, CV_8UC1);
                    cv::fillPoly(current_danger_roi_mask, std::vector<std::vector<cv::Point>>{danger_roi_polygon}, cv::Scalar(255));
                    prev_danger_roi_mask = current_danger_roi_mask.clone(); // 이전 ROI 마스크 저장
                } else {
                    current_danger_roi_mask = prev_danger_roi_mask.empty() ? cv::Mat::zeros(height, width, CV_8UC1) : prev_danger_roi_mask.clone();
                }

                if (!warning_roi_polygon.empty()) {
                    current_warning_roi_mask = cv::Mat::zeros(height, width, CV_8UC1);
                    cv::fillPoly(current_warning_roi_mask, std::vector<std::vector<cv::Point>>{warning_roi_polygon}, cv::Scalar(255));
                    
                    // Danger ROI 영역을 Warning ROI에서 제외 (Warning 영역이 Danger 영역과 겹치지 않도록)
                    cv::subtract(current_warning_roi_mask, current_danger_roi_mask, current_warning_roi_mask);
                    prev_warning_roi_mask = current_warning_roi_mask.clone(); // 이전 ROI 마스크 저장
                } else {
                    current_warning_roi_mask = prev_warning_roi_mask.empty() ? cv::Mat::zeros(height, width, CV_8UC1) : prev_warning_roi_mask.clone();
                }

            } else {
                // 이전 ROI 마스크 재사용
                current_danger_roi_mask = prev_danger_roi_mask.empty() ? cv::Mat::zeros(height, width, CV_8UC1) : prev_danger_roi_mask.clone();
                current_warning_roi_mask = prev_warning_roi_mask.empty() ? cv::Mat::zeros(height, width, CV_8UC1) : prev_warning_roi_mask.clone();
            }
            
            // 시각화용 프레임 복사
            cv::Mat roi_overlay = frame.clone(); // ROI를 그릴 투명 오버레이
            cv::Mat frame_output = frame.clone(); // YOLO 결과 및 FPS 등을 그릴 프레임

            // ROI 시각화 (빨강/노랑 폴리라인) - Python 코드에서 주석 처리된 부분이나, C++에서 활성화 가능
            /*
            if (!current_danger_roi_mask.empty()) {
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(current_danger_roi_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::polylines(roi_overlay, contours, true, cv::Scalar(0, 0, 255), 3); // Red
            }
            if (!current_warning_roi_mask.empty()) {
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(current_warning_roi_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::polylines(roi_overlay, contours, true, cv::Scalar(0, 255, 255), 3); // Yellow
            }
            */

            // 3. 객체 감지 및 ROI 필터링
            std::vector<DetectorResult> detections = detector.detect(frame);
            nlohmann::json danger_objects_json = nlohmann::json::array();

            for (const auto& det : detections) {
                int x1 = det.box.x;
                int y1 = det.box.y;
                int x2 = det.box.x + det.box.width;
                int y2 = det.box.y + det.box.height;
                std::string label = det.className;
                double distance = det.distance;

                // ROI 내부 확인
                bool in_danger = image_processor.isInsideROI(cv::Rect(x1, y1, det.box.width, det.box.height), current_danger_roi_mask, DANGER_THRESHOLD);
                bool in_warning = image_processor.isInsideROI(cv::Rect(x1, y1, det.box.width, det.box.height), current_warning_roi_mask, WARNING_THRESHOLD);

                // --- 서버 화면 디버깅용 시각화 (모든 객체 표시) ---
                // Python 코드에서 주석 처리된 부분. 필요 시 주석 해제하여 활성화
                /*
                cv::Scalar display_color_bgr;
                if (in_danger) display_color_bgr = cv::Scalar(0, 0, 255);       // Red
                else if (in_warning) display_color_bgr = cv::Scalar(0, 255, 255); // Yellow
                else display_color_bgr = cv::Scalar(0, 255, 0);                 // Green (Outside ROI)

                int box_thickness = 1;
                double label_font_scale = 0.7;
                int label_font_thickness = 1;
                double distance_font_scale = 0.5;
                int distance_font_thickness = 1;
                cv::Scalar shadow_color(0, 0, 0); // Black
                int shadow_offset = 1;

                cv::rectangle(frame_output, det.box, display_color_bgr, box_thickness);

                cv::Point label_text_pos(x1, y1 - 10);
                cv::putText(frame_output, label,
                            cv::Point(label_text_pos.x + shadow_offset, label_text_pos.y + shadow_offset),
                            cv::FONT_HERSHEY_SIMPLEX, label_font_scale, shadow_color, label_font_thickness + 1);
                cv::putText(frame_output, label, label_text_pos,
                            cv::FONT_HERSHEY_SIMPLEX, label_font_scale, display_color_bgr, label_font_thickness);

                cv::circle(frame_output, cv::Point(x1 + det.box.width / 2, y2), 5, display_color_bgr, -1);

                cv::Point distance_text_pos(x1, y2 + 20);
                std::string dist_text = "Dis: " + std::to_string(static_cast<int>(distance * 100 + 0.5) / 100.0) + "m";
                cv::putText(frame_output, dist_text,
                            cv::Point(distance_text_pos.x + shadow_offset, distance_text_pos.y + shadow_offset),
                            cv::FONT_HERSHEY_SIMPLEX, distance_font_scale, shadow_color, distance_font_thickness + 1);
                cv::putText(frame_output, dist_text, distance_text_pos,
                            cv::FONT_HERSHEY_SIMPLEX, distance_font_scale, display_color_bgr, distance_font_thickness);
                */

                // 클라이언트에 보낼 데이터 구성 (ROI 내 객체만)
                if (in_danger || in_warning) {
                    danger_objects_json.push_back({
                        {"label", label},
                        {"x", x1},
                        {"y", y1},
                        {"w", det.box.width},
                        {"h", det.box.height},
                        {"distance", std::round(distance * 100) / 100.0}, // 소수점 둘째자리 반올림
                        {"zone", in_danger ? "red" : "yellow"}
                    });
                }
            }

            // 4. FPS 계산 및 표시
            auto curr_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = curr_time - prev_time;
            prev_time = curr_time;
            fps = 1.0 / elapsed_seconds.count();

            cv::Point text_pos(10, 40);
            double font_scale = 0.8;
            int font_thickness = 2;
            cv::Scalar font_color(255, 255, 0); // BGR (하늘색)
            cv::Scalar shadow_color(0, 0, 0);   // BGR (검은색)
            int shadow_offset = 2;

            std::stringstream fps_ss;
            fps_ss << "FPS: " << std::fixed << std::setprecision(2) << fps;

            cv::putText(frame_output, fps_ss.str(), cv::Point(text_pos.x + shadow_offset, text_pos.y + shadow_offset),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, font_thickness);
            cv::putText(frame_output, fps_ss.str(), text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness);


            // 5. 클라이언트에게 응답 전송 (JSON 직렬화)
            std::string response_str = danger_objects_json.dump();
            try {
                server_socket.sendResponse(client_sock_fd, response_str);
            } catch (const std::runtime_error& e) {
                std::cerr << "ERROR: 응답 전송 중 오류: " << e.what() << std::endl;
                break; // 연결 끊김으로 간주하고 루프 종료
            }

            // 6. 디버깅 화면 출력 (오버레이 합성 후)
            cv::Mat final_display;
            cv::addWeighted(frame_output, 1.0, roi_overlay, 0.3, 0, final_display); // 투명 오버레이 적용

            cv::imshow("YOLO + Polygon Danger Detection", final_display);

            // 프레임 카운트 증가
            roi_frame_counter++; // ROI 업데이트 주기 계산용 카운터
            
            if (cv::waitKey(1) == 27) { // ESC 키로 종료
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "GENERAL SERVER ERROR: " << e.what() << std::endl;
    }

    // 자원 해제
    server_socket.stop(); // 소켓 닫기
    cv::destroyAllWindows();
    std::cout << "INFO: 서버가 종료되었습니다." << std::endl;

    return 0;
}