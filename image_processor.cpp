#include "image_processor.hpp"
#include <iostream>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::min, std::max
#include <opencv2/opencv.hpp> // 이 줄을 추가 또는 확인
#include <opencv2/core.hpp> 

ImageProcessor::ImageProcessor() {}

// OpenCV의 polyfit에 해당하는 기능 (단순화된 1차 근사)
// y를 독립 변수, x를 종속 변수로 사용 (수직선에 강함)
cv::Vec2f ImageProcessor::averageLine(const std::vector<cv::Vec4i>& lines, int height, int width) {
    if (lines.empty()) {
        return cv::Vec2f(0, 0); // 유효하지 않은 값 반환
    }

    std::vector<float> x_coords, y_coords;
    for (const auto& line : lines) {
        x_coords.push_back(line[0]);
        y_coords.push_back(line[1]);
        x_coords.push_back(line[2]);
        y_coords.push_back(line[3]);
    }

    // 최소 제곱법으로 1차 함수(x = my + c) 피팅
    // (Python의 np.polyfit(y, x, 1)과 유사)
    // C++에는 numpy.polyfit과 같은 내장 함수가 없으므로 직접 구현하거나 더 복잡한 라이브러리 사용
    // 여기서는 간단하게 평균을 이용한 근사를 시도합니다.
    // 더 정확한 피팅이 필요하면 Eigen 또는 GSL 같은 선형 대수 라이브러리를 고려해야 합니다.
    
    // 이 예시에서는 OpenCV의 fitLine을 사용하기 어려우므로, 간단한 통계적 근사 또는 수동 구현
    // 여기서는 간단히 y좌표의 평균과 x좌표의 평균을 사용하여 기울기를 추정하는 방식으로 대체.
    // 실제 np.polyfit처럼 정확하지는 않지만, 라인 검출 후 ROI 생성에는 충분할 수 있습니다.
    
    // x = my + c
    // y_mean, x_mean
    // m = sum((xi - x_mean)*(yi - y_mean)) / sum((yi - y_mean)^2)
    // c = x_mean - m * y_mean

    float sum_x = 0, sum_y = 0;
    for(size_t i = 0; i < x_coords.size(); ++i) {
        sum_x += x_coords[i];
        sum_y += y_coords[i];
    }
    float mean_x = sum_x / x_coords.size();
    float mean_y = sum_y / y_coords.size();

    float numerator = 0, denominator = 0;
    for(size_t i = 0; i < x_coords.size(); ++i) {
        numerator += (x_coords[i] - mean_x) * (y_coords[i] - mean_y);
        denominator += (y_coords[i] - mean_y) * (y_coords[i] - mean_y);
    }

    float m = 0;
    if (denominator != 0) {
        m = numerator / denominator;
    }
    float c = mean_x - m * mean_y;

    return cv::Vec2f(m, c); // m (기울기), c (y절편)
}


std::vector<cv::Point> ImageProcessor::createTrapezoidROI(const cv::Mat& frame, int y_bottom, int y_top) {
    int height = frame.rows;
    int width = frame.cols;

    cv::Mat gray, blur, edges;

    // 1. 그레이스케일 변환
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 2. 히스토그램 평활화 (선택적)
    // cv::equalizeHist(gray, gray); // Python 코드에서 주석 처리되어 있었음

    // 3. 가우시안 블러 (Python 코드와 동일하게 (3,3) 커널 사용)
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

    // 4. 캐니 에지 검출 (Python 코드와 동일하게 80, 150 임계값 사용)
    cv::Canny(blur, edges, 80, 150);

    // 5. 이전 에지 맵과 가중 평균하여 안정화 (Python 코드와 동일한 가중치)
    if (!prev_edges.empty()) {
        cv::addWeighted(edges, 0.7, prev_edges, 0.3, 0, edges);
    }
    prev_edges = edges.clone(); // 현재 에지 맵 저장

    // 관심영역 설정 (차선 검출을 위한 초기 마스크)
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Point roi_vertices_pts[] = {
        cv::Point(static_cast<int>(width * 0.1), height),
        cv::Point(static_cast<int>(width * 0.45), static_cast<int>(height * 0.4)),
        cv::Point(static_cast<int>(width * 0.55), static_cast<int>(height * 0.4)),
        cv::Point(static_cast<int>(width * 0.9), height)
    };
    const cv::Point* ppt[1] = { roi_vertices_pts };
    int npt[] = { 4 };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255));

    // ROI 내 차선 검출
    cv::Mat masked_edges;
    cv::bitwise_and(edges, mask, masked_edges);

    // 6. 확률적 허프 변환 (Python 코드와 동일한 파라미터 사용)
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(masked_edges, lines, 1, CV_PI / 180, 40, 30, 50); // image, lines (output), rho, theta, threshold, minLineLength, maxLineGap

    std::vector<cv::Vec4i> left_lines, right_lines;
    for (const auto& line : lines) {
        float slope = static_cast<float>(line[3] - line[1]) / (line[2] - line[0] + 1e-6f);
        if (slope < -0.5) { // 음수 기울기 (왼쪽 차선)
            left_lines.push_back(line);
        } else if (slope > 0.5) { // 양수 기울기 (오른쪽 차선)
            right_lines.push_back(line);
        }
    }

    // 7. 직선 근사로 평균화 (averageLine 함수 사용)
    cv::Vec2f left_fit = averageLine(left_lines, height, width);
    cv::Vec2f right_fit = averageLine(right_lines, height, width);

    // 유효성 검사 (기울기가 0이고 절편도 0이면 유효하지 않다고 판단)
    if ((left_fit[0] == 0 && left_fit[1] == 0) || (right_fit[0] == 0 && right_fit[1] == 0)) {
        return {}; // 빈 벡터 반환 (유효한 ROI를 생성할 수 없음)
    }

    // 감지된 차선을 바탕으로 ROI 폴리곤 정의
    int lx1 = static_cast<int>(left_fit[0] * y_bottom + left_fit[1]);
    int lx2 = static_cast<int>(left_fit[0] * y_top + left_fit[1]);
    int rx1 = static_cast<int>(right_fit[0] * y_bottom + right_fit[1]);
    int rx2 = static_cast<int>(right_fit[0] * y_top + right_fit[1]);

    // 차선 ROI가 화면 밖으로 나가지 않도록 클램핑
    lx1 = std::clamp(lx1, 0, width);
    lx2 = std::clamp(lx2, 0, width);
    rx1 = std::clamp(rx1, 0, width);
    rx2 = std::clamp(rx2, 0, width);

    // 유효한 좌표인지 최종 확인
    if (!(0 <= lx1 && lx1 <= width && 0 <= lx2 && lx2 <= width &&
          0 <= rx1 && rx1 <= width && 0 <= rx2 && rx2 <= width)) {
        return {}; // 유효하지 않은 ROI
    }

    std::vector<cv::Point> roi_polygon;
    roi_polygon.push_back(cv::Point(lx1, y_bottom));
    roi_polygon.push_back(cv::Point(lx2, y_top));
    roi_polygon.push_back(cv::Point(rx2, y_top));
    roi_polygon.push_back(cv::Point(rx1, y_bottom));

    return roi_polygon;
}

bool ImageProcessor::isInsideROI(const cv::Rect& bbox, const cv::Mat& roi_mask, float threshold) {
    if (roi_mask.empty() || bbox.width <= 0 || bbox.height <= 0) {
        return false;
    }

    // 바운딩 박스와 ROI 마스크의 교차 영역 추출
    cv::Rect intersection_rect = bbox & cv::Rect(0, 0, roi_mask.cols, roi_mask.rows);
    if (intersection_rect.empty()) {
        return false;
    }

    cv::Mat roi_box = roi_mask(intersection_rect);

    // 교차 영역 내에서 ROI 마스크 값이 255인 픽셀 수 계산
    int inside_pixels = cv::countNonZero(roi_box);

    // 바운딩 박스 영역 내에서 ROI에 포함된 비율 계산
    double coverage = static_cast<double>(inside_pixels) / (bbox.area() + 1e-6); // 0으로 나누기 방지
    
    return coverage >= threshold;
}