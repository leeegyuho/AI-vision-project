import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.publisher_ = self.create_publisher(Image, '/lgh/camera/image_raw', 10)
        self.bridge = CvBridge()

        # VIDEO_SOURCE는 실제 파일 경로로 변경해주세요
        self.video_source = '/home/hkit/Downloads/rural_cut.webm' # 실제 경로로 수정
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            self.get_logger().error(f"비디오 소스를 열 수 없습니다: {self.video_source}")
            raise IOError(f"Cannot open video source: {self.video_source}")

        self.resize_width = 640
        self.resize_height = 480

        self.timer_period = 1.0 / 30.0
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('Camera Publisher Node가 시작되었습니다.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("비디오 스트림의 끝에 도달했습니다. 재시작합니다.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("비디오를 다시 읽을 수 없습니다. 노드를 종료합니다.")
                self.destroy_node()
                rclpy.shutdown()
                return

        frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.publisher_.publish(ros_image)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraPublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except IOError as e:
        rclpy.logging.get_logger("camera_publisher_node").error(f"Initialization failed: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()