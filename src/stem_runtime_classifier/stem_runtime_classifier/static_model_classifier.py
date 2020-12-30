import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile


class StaticModelClassifier(Node):
    def __init__(self):
        super().__init__('stem_static_model_classifier')

        self.get_logger().info('Awaken.')


def main(args=None):
    rclpy.init(args=args)


if __name__ == '__main__':
    main()
