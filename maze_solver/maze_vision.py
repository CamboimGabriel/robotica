import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data

class MazeVisionNode(Node):
    def __init__(self):
        super().__init__('maze_vision_node')
        
        self.cam_sub = self.create_subscription(
            Image,
            '/front_camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data)
            
        self.color_pub = self.create_publisher(String, '/color_detected', 10)
        
        self.bridge = CvBridge()
        self.min_color_pixels = int(self.declare_parameter('min_color_pixels', 12000).value)
        self.last_published_color = "none"

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])
        lower_green = np.array([40, 100, 50])
        upper_green = np.array([80, 255, 255])
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        mask_red_1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask_red_2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        blue_pixels = cv2.countNonZero(mask_blue)
        
        msg_cor = String()
        pixel_counts = {
            "red": red_pixels,
            "green": green_pixels,
            "blue": blue_pixels,
        }
        dominant_color = max(pixel_counts, key=pixel_counts.get)
        if pixel_counts[dominant_color] >= self.min_color_pixels:
            msg_cor.data = dominant_color
        else:
            msg_cor.data = "none"

        self.color_pub.publish(msg_cor)

        if msg_cor.data != self.last_published_color:
            self.get_logger().info(
                f"[COR {self.last_published_color} -> {msg_cor.data}] "
                f"vermelho={red_pixels} verde={green_pixels} azul={blue_pixels} "
                f"limite={self.min_color_pixels}"
            )
            self.last_published_color = msg_cor.data

        mask_total = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, mask_blue))
        cv2.imshow("Robot vision", mask_total)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = MazeVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
