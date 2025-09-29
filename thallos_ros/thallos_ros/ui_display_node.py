import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ros_interfaces.msg import ProcessingData, DetectedObject # 💡 메시지 임포트 경로 변경
from cv_bridge import CvBridge
import cv2
import numpy as np

class UIDisplayNode(Node):
    def __init__(self):
        super().__init__('ui_display_node')
        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(
            Image,
            '/processing/visualized_image',
            self.image_callback,
            10
        )

        self.data_subscription = self.create_subscription(
            ProcessingData,
            '/processing/data_output',
            self.data_callback,
            10
        )

        self.latest_processed_data = None
        self.get_logger().info('UI Display Node가 시작되었습니다.')

        cv2.namedWindow("Client View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Client View", 640, 480)

    def data_callback(self, msg: ProcessingData):
        self.latest_processed_data = msg

    def image_callback(self, img_msg: Image):
        if self.latest_processed_data is None:
            self.get_logger().warn("알림 레벨 데이터가 아직 수신되지 않았습니다. 대기 중...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            #alert_text = self.latest_processed_data.alert_level
            #alert_color = (255, 255, 255)
            #if alert_text == "WARNING":
            #    alert_color = (0, 255, 255)
            #elif alert_text == "DANGER":
            #    alert_color = (0, 0, 255)

            #if alert_text != "NONE":
            #    font_scale = 1.5
            #    thickness = 3
            #    text_size, baseline = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            #    text_x = int((cv_image.shape[1] - text_size[0]) / 2)
            #    text_y = text_size[1] + 20

            #    cv2.putText(cv_image, alert_text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
            #    cv2.putText(cv_image, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, alert_color, thickness)

            current_y_offset = 70
            line_height = 75
            box_width = 250
            box_x = 0

            

            for i, obj in enumerate(self.latest_processed_data.detected_objects):
                label = obj.label
                dist = obj.distance
                zone = obj.zone

                y_top = current_y_offset + i * line_height
                x_center = box_x + box_width // 2
                
                
                if zone == "red":
                    triangle_pts = np.array([
                        [x_center, y_top + 10],
                        [box_x + 10, y_top + line_height - 10],
                        [box_x + box_width - 10, y_top + line_height - 10]
                    ], np.int32)
                    cv2.fillPoly(cv_image, [triangle_pts], color=(0, 0, 255))

                    # danger_text = "DANGER" # 이 부분은 여전히 주석 처리된 상태로 둡니다.
                    # dz_size, _ = cv2.getTextSize(danger_text, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)
                    # dz_x = x_center - dz_size[0] // 2
                    # dz_y = triangle_pts[0][1] - 8
                    # #cv2.putText(cv_image, danger_text, (dz_x, dz_y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 3)

                    content_text = f"DANGER" # 이 텍스트를 중앙에 위치시킵니다.
                    ct_size, ct_baseline = cv2.getTextSize(content_text, cv2.FONT_HERSHEY_PLAIN, 1.5, 1) # 폰트 크기 1.5, 두께 2에 맞춰 재측정
                    
                    # 삼각형 중앙 Y 좌표 계산 (삼각형 꼭짓점들의 평균 Y 값)
                    triangle_center_y = (triangle_pts[0][1] + triangle_pts[1][1] + triangle_pts[2][1]) // 3

                    # 텍스트의 수직 중앙을 삼각형 중앙에 맞추기
                    # cv2.putText는 텍스트의 왼쪽 하단(baseline)을 기준으로 그리므로 조정이 필요합니다.
                    ct_x = x_center - ct_size[0] // 2 # 수평 중앙
                    ct_y = triangle_center_y + (ct_size[1] // 2) - (ct_baseline // 2) # 수직 중앙 정렬 (텍스트 높이 및 기준선 고려)

                    # 1. 검은색 윤곽선 그리기 (원본 텍스트보다 살짝 두껍게)
                    outline_thickness = 3 # 예를 들어, 원래 두께가 2였으므로 2를 더해서 4로 설정
                    cv2.putText(cv_image, content_text, (ct_x, ct_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), outline_thickness)


                    cv2.putText(cv_image, content_text, (ct_x, ct_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                
                elif zone == "yellow":
                    y_center_diamond = y_top + line_height // 2
                    diamond_pts = np.array([
                        [x_center, y_top + 10],
                        [box_x + box_width - 10, y_center_diamond],
                        [x_center, y_top + line_height - 10],
                        [box_x + 10, y_center_diamond]
                    ], np.int32)
                    cv2.fillPoly(cv_image, [diamond_pts], color=(0, 255, 255))

                    # warn_text = "WARNING" # 이 부분은 여전히 주석 처리된 상태로 둡니다.
                    # wz_size, _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)
                    # wz_x = x_center - wz_size[0] // 2 + 10
                    # wz_y = diamond_pts[0][1] - 8
                    # #cv2.putText(cv_image, warn_text, (wz_x, wz_y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 3)

                    content_text = f"WARNING" # 이 텍스트를 중앙에 위치시킵니다.
                    ct_size, ct_baseline = cv2.getTextSize(content_text, cv2.FONT_HERSHEY_PLAIN, 1.5, 1) # 폰트 크기 1.5, 두께 2에 맞춰 재측정
                    
                    # 텍스트의 수직 중앙을 마름모 중앙(y_center_diamond)에 맞추기
                    ct_x = x_center - ct_size[0] // 2 # 수평 중앙
                    ct_y = y_center_diamond + (ct_size[1] // 2) - (ct_baseline // 2) # 수직 중앙 정렬

                    # 1. 검은색 윤곽선 그리기 (원본 텍스트보다 살짝 두껍게)
                    outline_thickness = 3 # 예를 들어, 원래 두께가 2였으므로 2를 더해서 4로 설정
                    cv2.putText(cv_image, content_text, (ct_x, ct_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), outline_thickness)

                    cv2.putText(cv_image, content_text, (ct_x, ct_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                    

                    

            cv2.imshow("Client View", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"UI Node 이미지 처리 오류: {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UIDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()