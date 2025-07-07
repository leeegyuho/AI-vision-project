#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <opencv2/core.hpp> 

class ImageProcessor {
public:
    ImageProcessor();

    // 트랩 형태의 ROI를 생성하고 반환 (OpenCV 폴리곤 점 목록)
    std::vector<cv::Point> createTrapezoidROI(const cv::Mat& frame, int y_bottom, int y_top);

    // 바운딩 박스가 ROI 마스크 내에 일정 비율 이상 포함되었는지 확인
    bool isInsideROI(const cv::Rect& bbox, const cv::Mat& roi_mask, float threshold);

private:
    cv::Mat prev_edges; // createTrapezoidROI에서 이전 에지 맵을 저장

    // 직선 근사로 평균화 (createTrapezoidROI 내부에서 사용)
    cv::Vec2f averageLine(const std::vector<cv::Vec4i>& lines, int height, int width);
};

#endif // IMAGE_PROCESSOR_HPP