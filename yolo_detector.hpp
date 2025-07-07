#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>
#include <map>

// 감지 결과를 저장할 구조체
struct DetectorResult {
    cv::Rect box;
    float confidence;
    std::string className;
    double distance;
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_xml_path, const std::string& model_bin_path,
                 const std::string& device, float conf_threshold, float nms_threshold);
    ~YoloDetector();

    bool isInitialized() const { return initialized; }
    std::vector<DetectorResult> detect(const cv::Mat& frame);

private:
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    bool initialized;

    float conf_threshold;
    float nms_threshold;

    // COCO 데이터셋 80개 클래스 이름 (여러분의 모델 클래스에 맞춰 수정 필요)
    std::map<int, std::string> class_names = {
        {0, "vehicle"}, {1, "bigvehicle"}, {2, "bike"}, {3, "human"}, {4, "animal"},
        {5, "obstacle"}
    };

    // 거리 계산에 사용되는 상수들 (Python 코드와 동일)
    const float FOCAL_LENGTH = 600.0f;

    std::map<std::string, float> REAL_HEIGHTS = {
        {"person", 1.6f}, {"car", 1.5f}, {"bus", 3.2f}, {"truck", 3.4f},
        {"motorbike", 1.4f}, {"bicycle", 1.2f}, {"vehicle", 1.5f},
        {"big vehicle", 3.5f}, {"bike", 1.2f}, {"human", 1.7f},
        {"animal", 0.5f}, {"obstacle", 1.0f}
    };

    std::map<std::string, float> REAL_WIDTHS = {
        {"person", 0.5f}, {"car", 1.8f}, {"bus", 2.5f}, {"truck", 2.5f},
        {"motorbike", 0.8f}, {"bicycle", 0.7f}, {"vehicle", 1.8f},
        {"big vehicle", 2.5f}, {"bike", 0.5f}, {"human", 0.5f},
        {"animal", 0.6f}, {"obstacle", 1.0f}
    };

    double estimateDistance(float h, float w, const std::string& label);
};

#endif // YOLO_DETECTOR_HPP