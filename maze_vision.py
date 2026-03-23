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

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Color limiar (HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_green = np.array([40, 100, 50])
        upper_green = np.array([80, 255, 255])
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        blue_pixels = cv2.countNonZero(mask_blue)

        limit_pixels = 50000 
        
        msg_cor = String()

        if red_pixels > limit_pixels:
            msg_cor.data = "red"
        elif green_pixels > limit_pixels:
            msg_cor.data = "green"
        elif blue_pixels > limit_pixels:
            msg_cor.data = "blue"
        else:
            msg_cor.data = "none"

        # to the state_machine
        self.color_pub.publish(msg_cor)
        
        self.get_logger().info(f'Color: {msg_cor.data}')

        mask_total = mask_red + mask_green + mask_blue
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
