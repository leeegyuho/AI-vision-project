import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ros_interfaces.msg import ProcessingData, DetectedObject # 💡 메시지 임포트 경로 변경
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# --- 설정 (이전과 동일) ---
FOCAL_LENGTH = 600

REAL_HEIGHTS = {
    "person": 1.6, "car": 1.5, "bus": 3.2, "truck": 3.4,
    "motorbike": 1.4, "bicycle": 1.2, "vehicle": 1.5,
    "big vehicle": 3.5, "bike": 1.2, "human": 1.7,
    "animal": 0.5, "obstacle": 1.0
}

REAL_WIDTHS = {
    "person": 0.5, "car": 1.8, "bus": 2.5, "truck": 2.5,
    "motorbike": 0.8, "bicycle": 0.7, "vehicle": 1.8,
    "big vehicle": 2.5, "bike": 0.5, "human": 0.5,
    "animal": 0.6, "obstacle": 1.0
}

WARNING_DISTANCE_THRESHOLD = 15.0
DANGER_DISTANCE_THRESHOLD = 5.0

danger_threshold = 0.1
warning_threshold = 0.1

class ObjectDetectionProcessingNode(Node):
    def __init__(self):
        super().__init__('object_detection_processing_node')
        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(
            Image,
            '/lgh/camera/image_raw',
            self.image_callback,
            10
        )

        self.yolo_model_path = '/home/hkit/Desktop/model test result/yolov8_custom14_test 7_n_250625/weights/best.pt' # 실제 경로로 수정
        try:
            # self.yolo_model = YOLO(self.yolo_model_path).to('cuda')
            self.yolo_model = YOLO(self.yolo_model_path)
            self.get_logger().info(f"YOLOv8 모델 로드 성공: {self.yolo_model_path}")
        except Exception as e:
            self.get_logger().error(f"YOLOv8 모델 로드 실패: {e}")
            self.yolo_model = None

        self.processed_data_publisher_ = self.create_publisher(ProcessingData, '/processing/data_output', 10)
        #self.visualized_image_publisher_ = self.create_publisher(Image, '/processing/visualized_image', 10)

        self.get_logger().info('Object Detection Processing Node가 시작되었습니다.')

        self.prev_frame_time = time.time()
        self.prev_edges = None
        self.frame_count = 0
        self.roi_update_interval = 5
        self.prev_danger_roi = None
        self.prev_warning_roi = None

        self.class_names = self.yolo_model.names if self.yolo_model else {}

    def estimate_distance(self, h, w, label):
        try:
            dist_h = (REAL_HEIGHTS[label] * FOCAL_LENGTH) / h
            dist_w = (REAL_WIDTHS[label] * FOCAL_LENGTH) / w
            return (dist_h + dist_w) / 2
        except KeyError:
            return -1.0

    def inside_roi(self, box, mask, threshold):
        x1, y1, x2, y2 = map(int, box)
        roi_box = mask[y1:y2, x1:x2]
        if roi_box.size == 0:
            return False
        inside = np.count_nonzero(roi_box == 255)
        return inside / roi_box.size >= threshold

    def create_trapezoid_roi(self, frame, y_bottom, y_top):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 140)

        if self.prev_edges is not None:
            edges = cv2.addWeighted(edges.astype(np.float32), 0.7,
                                     self.prev_edges.astype(np.float32), 0.3, 0).astype(np.uint8)
        self.prev_edges = edges.copy()

        roi_vertices = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.45), int(height * 0.5)),
            (int(width * 0.55), int(height * 0.5)),
            (int(width * 0.9), height)
        ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=70)

        left_lines, right_lines = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < -0.5: left_lines.append((x1, y1, x2, y2))
                elif slope > 0.5: right_lines.append((x1, y1, x2, y2))

        def average_line(lines_coords):
            if not lines_coords: return None
            x, y = [], []
            for x1, y1, x2, y2 in lines_coords:
                x += [x1, x2]; y += [y1, y2]
            if len(x) < 2: return None
            return np.polyfit(y, x, 1)

        left_fit = average_line(left_lines)
        right_fit = average_line(right_lines)

        if left_fit is None or right_fit is None:
            return None

        lx1, lx2 = int(np.polyval(left_fit, y_bottom)), int(np.polyval(left_fit, y_top))
        rx1, rx2 = int(np.polyval(right_fit, y_bottom)), int(np.polyval(right_fit, y_top))

        lx1 = np.clip(lx1, 0, width)
        lx2 = np.clip(lx2, 0, width)
        rx1 = np.clip(rx1, 0, width)
        rx2 = np.clip(rx2, 0, width)

        if not (0 <= lx1 <= width and 0 <= lx2 <= width and
                0 <= rx1 <= width and 0 <= rx2 <= width):
            return None

        return np.array([[(lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)]], dtype=np.int32)

    def image_callback(self, msg):
        self.frame_count += 1

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 오류: {e}")
            return

        h, w = cv_image.shape[:2]
        debug_image = cv_image.copy()

        dynamic_danger_bottom = int(h * 1)
        dynamic_danger_top = int(h * 0.8)

        dynamic_warning_bottom = int(h * 0.8)
        dynamic_warning_top = int(h * 0.6)

        if self.frame_count % self.roi_update_interval == 0:
            danger_roi = self.create_trapezoid_roi(cv_image, dynamic_danger_bottom, dynamic_danger_top)
            warning_roi = self.create_trapezoid_roi(cv_image, dynamic_warning_bottom, dynamic_warning_top)
            if danger_roi is not None:
                self.prev_danger_roi = danger_roi
            if warning_roi is not None:
                self.prev_warning_roi = warning_roi
        else:
            danger_roi = self.prev_danger_roi
            warning_roi = self.prev_warning_roi

        mask_danger = np.zeros((h, w), dtype=np.uint8)
        mask_warning = np.zeros((h, w), dtype=np.uint8)

        if danger_roi is not None:
            cv2.fillPoly(mask_danger, [danger_roi], 255)
        if warning_roi is not None:
            cv2.fillPoly(mask_warning, [warning_roi], 255)
            mask_warning = cv2.subtract(mask_warning, mask_danger)

        if danger_roi is not None:
            cv2.polylines(debug_image, danger_roi, isClosed=True, color=(0, 0, 255), thickness=2)
        if warning_roi is not None:
            warning_mask_for_viz = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(warning_mask_for_viz, [warning_roi], 255)
            danger_mask_for_viz = np.zeros((h, w), dtype=np.uint8)
            if danger_roi is not None:
                cv2.fillPoly(danger_mask_for_viz, [danger_roi], 255)
            warning_mask_for_viz = cv2.subtract(warning_mask_for_viz, danger_mask_for_viz)

            contours, _ = cv2.findContours(warning_mask_for_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.polylines(debug_image, contours, isClosed=True, color=(0, 255, 255), thickness=2)

        detected_objects_for_msg = []
        current_alert_level = "NONE"

        if self.yolo_model:
            results = self.yolo_model(cv_image, verbose=False)
            for r in results:
                if r.boxes is not None and r.boxes.xyxy is not None:
                    for *xyxy, conf, cls in r.boxes.data:
                        x1, y1, x2, y2 = map(int, xyxy)
                        label_id = int(cls)
                        label = self.class_names.get(label_id, f"Unknown({label_id})")

                        bbox_w, bbox_h = x2 - x1, y2 - y1
                        if bbox_w <= 0 or bbox_h <= 0:
                            continue

                        distance = self.estimate_distance(bbox_h, bbox_w, label)

                        obj_center_x = int((x1 + x2) / 2)
                        obj_center_y = int(y2)
                        object_point = (obj_center_x, obj_center_y)

                        in_danger = self.inside_roi((x1, y1, x2, y2), mask_danger, danger_threshold)
                        in_warning = self.inside_roi((x1, y1, x2, y2), mask_warning, warning_threshold)

                        zone_str = "none"
                        if in_danger:
                            zone_str = "red"
                            if distance < DANGER_DISTANCE_THRESHOLD:
                                current_alert_level = "DANGER"
                        elif in_warning:
                            zone_str = "yellow"
                            if distance < WARNING_DISTANCE_THRESHOLD and current_alert_level != "DANGER":
                                current_alert_level = "WARNING"

                        # === 이 부분이 핵심 수정 지점입니다. ===
                        # 객체가 danger_roi 또는 warning_roi 내에 있을 경우에만 메시지에 추가
                        if in_danger or in_warning: # 또는 원하는 조건으로 변경 (예: if in_danger:)
                            display_color_bgr = (0, 0, 255) if in_danger else \
                                                (0, 255, 255) if in_warning else \
                                                (0, 0, 0) # 이 부분은 이제 필요 없을 수도 있습니다 (else로 들어오지 않으므로)

                            cv2.rectangle(debug_image, (x1, y1), (x2, y2), display_color_bgr, 2)
                            label_text = f"{label} {distance:.2f}m"
                            cv2.putText(debug_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, display_color_bgr, 1)
                            cv2.circle(debug_image, (obj_center_x, obj_center_y), 5, display_color_bgr, -1)

                        detected_obj_msg = DetectedObject()
                        detected_obj_msg.label = label
                        detected_obj_msg.distance = float(distance)
                        detected_obj_msg.zone = zone_str
                        detected_obj_msg.x_min = x1
                        detected_obj_msg.y_min = y1
                        detected_obj_msg.width = bbox_w
                        detected_obj_msg.height = bbox_h
                        detected_objects_for_msg.append(detected_obj_msg)

        processed_data_msg = ProcessingData()
        processed_data_msg.header = msg.header
        processed_data_msg.image_width = w
        processed_data_msg.image_height = h
        processed_data_msg.alert_level = current_alert_level
        processed_data_msg.detected_objects = detected_objects_for_msg

       #processed_data_msg.danger_roi_available = (danger_roi is not None)
        if danger_roi is not None:
            processed_data_msg.danger_roi_x_coords = [int(p[0]) for p in danger_roi[0]]
            processed_data_msg.danger_roi_y_coords = [int(p[1]) for p in danger_roi[0]]

        #processed_data_msg.warning_roi_available = (warning_roi is not None)
        if warning_roi is not None:
            processed_data_msg.warning_roi_x_coords = [int(p[0]) for p in warning_roi[0]]
            processed_data_msg.warning_roi_y_coords = [int(p[1]) for p in warning_roi[0]]

        self.processed_data_publisher_.publish(processed_data_msg)

        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_frame_time)
        self.prev_frame_time = curr_time
        cv2.putText(debug_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        visualized_ros_image = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        self.visualized_image_publisher_.publish(visualized_ros_image)

        cv2.imshow("Processing Node Output", debug_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionProcessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()