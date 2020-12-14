

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from collections import deque

from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal


from stem_lib.stdlib.stopwatch import Stopwatch
from stem_lib import utils as stem_utils
from stem_lib import learning_utils



class StemStateSampler(Node):
    def __init__(self):
        super().__init__('stem_state_sampler')

        self.get_logger().info('Awaken.')

        self.working_dir = stem_utils.load_parameter(self, 'working_dir', '.stem/' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        self.sensor_data_queue_size = stem_utils.load_parameter(self, 'sensor_data_queue_size', 100)
        self.state_names = stem_utils.load_parameter(self, 'state_names', ['touched', 'not_touched'])
        self.sensor_data_segment_size = stem_utils.load_parameter(self, 'sensor_data_segment_size', 2)
        self.sensor_sampling_rate_min = stem_utils.load_parameter(self, 'sensor_sampling_rate_min', 35)

        self.sensor_receiver = self.create_subscription(
            GeneralSensorData,
            'general_sensor_data',
            self.on_receive_sensor_data,
            QoSProfile(depth=10)
        )
        
        self.supervise_signal_receiver = self.create_subscription(
            SuperviseSignal,
            'supervise_signal',
            self.on_receive_supervise_signal,
            QoSProfile(depth=10)
        )

        self.sensor_data_sampleing_sw = Stopwatch()
        self.sensor_data_queue = deque(maxlen=self.sensor_data_queue_size)

    def start(self):
        self.sensor_data_sampleing_sw.start()

    def on_receive_sensor_data(self, sensor_data):
    
    def on_receive_supervise_signal(self, supervise_signal):
        


def main(args=None):
    rclpy.init(args=args)

    stem_state_sampler = StemStateSampler()
    rclpy.spin(stem_state_sampler)

    stem_state_sampler.destroy_node()
    stem_state_sampler.shutdown()

if __name__ == '__main__':
    main()