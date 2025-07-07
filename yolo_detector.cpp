#include "yolo_detector.hpp"
#include <iostream>
#include <numeric> // For std::iota
#include <algorithm> // For std::sort

YoloDetector::YoloDetector(const std::string& model_xml_path, const std::string& model_bin_path,
                           const std::string& device, float conf_threshold, float nms_threshold)
    : initialized(false), conf_threshold(conf_threshold), nms_threshold(nms_threshold) {
    try {
        // 모델 로드
        std::shared_ptr<ov::Model> model = core.read_model(model_xml_path);

        // 입력/출력 설정 확인
        // YOLOv8 모델은 일반적으로 1개의 입력과 1개의 출력을 가집니다.
        // 입력: [1, 3, H, W] (H, W는 모델의 입력 해상도)
        // 출력: [1, BoundingBox_Data, Num_Classes + 4(bbox)] 또는 [1, Num_Classes + 4(bbox), BoundingBox_Data]
        
        ov::Output<const ov::Node> input_port = model->input();
        ov::Output<const ov::Node> output_port = model->output();

        // 입력 리사이즈 모드 설정 (LETTERBOX)
        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32); // 출력 타입 float32

        model = ppp.build();

        // 모델 컴파일
        compiled_model = core.compile_model(model, device);
        infer_request = compiled_model.create_infer_request();

        initialized = true;
        std::cout << "INFO: YOLO Detector 초기화 완료 (OpenVINO)." << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "ERROR: YOLO Detector 초기화 중 오류 발생: " << ex.what() << std::endl;
        initialized = false;
    }
}

YoloDetector::~YoloDetector() {
    // 소멸자에서 특별히 해제할 리소스는 OpenVINO 객체가 자동으로 관리합니다.
}

std::vector<DetectorResult> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<DetectorResult> results;
    if (!initialized) {
        std::cerr << "WARNING: Detector가 초기화되지 않았습니다." << std::endl;
        return results;
    }

    int input_height = compiled_model.input().get_shape()[2];
    int input_width = compiled_model.input().get_shape()[3];

    cv::Mat processed_frame;
    // 입력 모델의 해상도에 맞춰 리사이즈
    cv::resize(frame, processed_frame, cv::Size(input_width, input_height));

    // OpenCV Mat을 OpenVINO Tensor로 변환
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), 
                                         compiled_model.input().get_shape(), 
                                         processed_frame.data);
    infer_request.set_input_tensor(input_tensor);

    // 추론 수행
    infer_request.infer();

    // 결과 가져오기
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    
    // OpenVINO 모델 출력 형태에 따라 후처리 로직이 달라집니다.
    // YOLOv8 OpenVINO IR 모델의 일반적인 출력 형태는 [1, N, 84] (N은 감지된 박스 수, 84는 [x,y,w,h, conf, 80 classes]) 입니다.
    // 또는 [1, 84, N] 형태일 수도 있습니다. 모델의 shape을 확인해야 합니다.
    
    const float* output_data = output_tensor.data<const float>();
    
    // YOLOv8 ONNX/OpenVINO 모델의 일반적인 출력 형태는 [1, 84, 8400] (8400은 앵커 박스 수) 또는 [1, 8400, 84] 입니다.
    // 여기서 84는 (x, y, w, h, confidence, class_scores...) 입니다.
    // 출력 shape을 기준으로 파싱합니다.
    ov::Shape output_shape = output_tensor.get_shape();
    
    int num_boxes = 0;
    int data_offset = 0; // 각 박스 데이터의 시작 오프셋 (confidence + class_scores)
    
    // output_shape: [1, 8400, 84] or [1, 84, 8400]
    // 여기서 num_boxes는 8400, data_offset은 84 (x,y,w,h,conf,class_probs...)
    // 또는 num_boxes는 84, data_offset은 8400
    
    // OpenVINO에서 export된 YOLOv8은 보통 [1, Num_features, Num_boxes] 형태입니다.
    // Num_features = 4 (bbox) + 1 (conf) + 80 (classes) = 85
    // 그래서 보통 [1, 85, N] 형태입니다.
    
    if (output_shape.size() == 3) { // [1, Num_features, Num_boxes] 또는 [1, Num_boxes, Num_features]
        // YOLOv8의 경우, 일반적으로 [1, Num_features(85), Num_boxes(8400)] 형태입니다.
        if (output_shape[1] == 84 || output_shape[1] == 85) { // 84 또는 85 특징 (xywh + conf + 80 classes)
            num_boxes = output_shape[2]; // N
            data_offset = output_shape[1]; // 84 또는 85
        } else if (output_shape[2] == 84 || output_shape[2] == 85) { // 84 또는 85 특징
            num_boxes = output_shape[1]; // N
            data_offset = output_shape[2]; // 84 또는 85
        } else {
             std::cerr << "WARNING: 예상치 못한 YOLOv8 출력 shape: [" << output_shape[0] << ", " << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
             return results;
        }
    } else {
        std::cerr << "WARNING: 예상치 못한 YOLOv8 출력 텐서 차원: " << output_shape.size() << std::endl;
        return results;
    }

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // Output parsing (YOLOv8 specific)
    for (int i = 0; i < num_boxes; ++i) {
        float x_center, y_center, w, h;
        float confidence;
        int class_id = -1;
        float max_class_score = 0.0f;

        // Assuming output is [1, 85, N] (transposed from [1, N, 85])
        // If it's [1, N, 85], then output_data + i * data_offset for each box
        // If it's [1, 85, N], then output_data + j * num_boxes + i for feature j of box i

        if (output_shape[1] == data_offset) { // [1, features, boxes] (transposed)
            x_center = output_data[0 * num_boxes + i];
            y_center = output_data[1 * num_boxes + i];
            w = output_data[2 * num_boxes + i];
            h = output_data[3 * num_boxes + i];
            confidence = output_data[4 * num_boxes + i]; // Objectness score
            
            // 클래스 스코어 (80개 클래스)
            for (int j = 0; j < (data_offset - 5); ++j) { // 5는 x,y,w,h,conf
                float class_score = output_data[(5 + j) * num_boxes + i];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    class_id = j;
                }
            }
        } else { // [1, boxes, features]
            const float* box_data = output_data + i * data_offset;
            x_center = box_data[0];
            y_center = box_data[1];
            w = box_data[2];
            h = box_data[3];
            confidence = box_data[4]; // Objectness score

            // 클래스 스코어
            for (int j = 0; j < (data_offset - 5); ++j) {
                float class_score = box_data[5 + j];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    class_id = j;
                }
            }
        }

        // 최종 신뢰도 계산 (objectness * class_score)
        float final_confidence = confidence * max_class_score;

        if (final_confidence > conf_threshold) {
            float x = (x_center - w / 2.0f);
            float y = (y_center - h / 2.0f);

            // 모델의 입력 해상도에서 원본 프레임 해상도로 스케일링
            float x_scale = static_cast<float>(frame.cols) / input_width;
            float y_scale = static_cast<float>(frame.rows) / input_height;

            cv::Rect bbox(static_cast<int>(x * x_scale),
                          static_cast<int>(y * y_scale),
                          static_cast<int>(w * x_scale),
                          static_cast<int>(h * y_scale));

            bboxes.push_back(bbox);
            confidences.push_back(final_confidence);
            class_ids.push_back(class_id);
        }
    }

    // NMS (Non-Maximum Suppression) 적용
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        std::string class_name = "unknown";
        if (class_names.count(class_ids[idx])) {
            class_name = class_names[class_ids[idx]];
        }
        
        // 거리 계산
        double distance = estimateDistance(bboxes[idx].height, bboxes[idx].width, class_name);

        results.push_back({bboxes[idx], confidences[idx], class_name, distance});
    }

    return results;
}

double YoloDetector::estimateDistance(float h, float w, const std::string& label) {
    try {
        if (REAL_HEIGHTS.count(label) && REAL_WIDTHS.count(label)) {
            double dist_h = (REAL_HEIGHTS[label] * FOCAL_LENGTH) / h;
            double dist_w = (REAL_WIDTHS[label] * FOCAL_LENGTH) / w;
            return (dist_h + dist_w) / 2.0;
        } else {
            return -1.0; // 정의되지 않은 라벨
        }
    } catch (const std::exception& e) {
        std::cerr << "WARNING: 거리 추정 중 오류: " << e.what() << std::endl;
        return -1.0;
    }
}