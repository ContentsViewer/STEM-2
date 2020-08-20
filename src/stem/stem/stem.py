import rclpy
from rclpy.node import Node

from stem_interfaces.msg import GeneralSensorData

class STEM(Node):
    def __init__(self):
        super().__init__('stem')
        self.sensor_receiver = self.create_subscription(
            GeneralSensorData,
            'general_sensor_data',
            self.on_receive_sensor_data,
            10)
        
    def on_receive_sensor_data(self, sensor_data):
        # self.get_logger().info(str(sensor_data.segments))
        pass

def main(args=None):
    rclpy.init(args=args)

    stem = STEM()

    rclpy.spin(stem)

    stem.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()