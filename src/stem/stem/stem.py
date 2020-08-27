import os
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal

from stem_lib.stdlib import runtime_resources
from stem_lib.stdlib.bimapper import BiMapper
from stem_lib import utils as stem_utils
from stem_lib import learning_utils

class STEM(Node):
    def __init__(self):
        super().__init__('stem')

        self.sensor_data_queue_size = stem_utils.load_parameter(self, "sensor_data_queue_size", 100)
        self.state_name_list = stem_utils.load_parameter(self, "state_name_list", ["touched", "not_touched"])
        self.sensor_data_segment_count = stem_utils.load_parameter(self, "sensor_data_segment_count", 2)
        
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

        self.sensor_data_queue = deque(maxlen=self.sensor_data_queue_size)
        self.model = learning_utils.make_model(
            self.sensor_data_queue_size, self.sensor_data_segment_count
        )

        resources = runtime_resources.Resources('.stem/')


    def on_receive_sensor_data(self, sensor_data):
        if len(sensor_data.segments) != self.sensor_data_segment_count:
            self.get_logger().warning('sensor_data segment count is incompatible.')
            return
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