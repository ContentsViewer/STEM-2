import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal

from collections import deque

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

        self.sensor_data_queue = deque(maxlen=100)

    def on_receive_sensor_data(self, sensor_data):
        # self.get_logger().info(str(sensor_data.segments))

        self.sensor_data_queue.append(sensor_data.segments)
    
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