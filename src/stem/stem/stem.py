import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal

import tensorflow as tf

class STEM(Node):
    def __init__(self):
        super().__init__('stem')
        self.sensor_receiver = self.create_subscription(
            GeneralSensorData,
            'general_sensor_data',
            self.on_receive_sensor_data,
            QoSProfile(depth=10)
        )
        
        self.supervise_signal_receiver = self.create_subscription(
            SuperviseSignal,
            'supervise_signal',
            self.on_receiver_supervise_signal,
            QoSProfile(depth=10)
        )

        
    def on_receive_sensor_data(self, sensor_data):
        # self.get_logger().info(str(sensor_data.segments))
        pass
    
    def on_receiver_supervise_signal(self, supervise_signal):
        # self.get_logger().info(supervise_signal.supervised_state_name)
        pass


def main(args=None):
    rclpy.init(args=args)

    stem = STEM()

    rclpy.spin(stem)

    stem.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()