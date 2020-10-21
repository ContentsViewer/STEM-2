import os
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
import numpy as np

from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal
from stem_interfaces.msg import STEMStatus

from stem_lib.stdlib import runtime_resources
from stem_lib.stdlib.bimapper import BiMapper
from stem_lib.stdlib.threading import ThreadWorker
from stem_lib.stdlib.stopwatch import Stopwatch
from stem_lib import utils as stem_utils
from stem_lib import learning_utils

class STEM(Node):
    def __init__(self):
        super().__init__('stem')

        self.get_logger().info('STEM is Awaken!')

        self.sensor_data_queue_size = stem_utils.load_parameter(self, 'sensor_data_queue_size', 100)
        self.state_name_list = stem_utils.load_parameter(self, 'state_name_list', ['touched', 'not_touched'])
        self.sensor_data_segment_size = stem_utils.load_parameter(self, 'sensor_data_segment_size', 2)
        self.replay_buffer_maxlen = stem_utils.load_parameter(self, 'replay_buffer_maxlen', 100)
        self.nmin_samples_replay_buffer = stem_utils.load_parameter(self, 'nmin_samples_replay_buffer', 50)
        
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

        self.status_publisher = self.create_publisher(
            STEMStatus,
            'stem_status',
            QoSProfile(depth=10))

        self.status = {
            'sensor_data_queue_length': 0
        }

        self.sensor_data_sampleing_sw = Stopwatch()

        self.timer = self.create_timer(0.1, self.test)

        self.sensor_data_queue = deque(maxlen=self.sensor_data_queue_size)
        self.model = learning_utils.make_model(
            self.sensor_data_queue_size, self.sensor_data_segment_size
        )

        self.replay_buffer = learning_utils.ReplayBuffer(len(self.state_name_list), self.replay_buffer_maxlen)
        self.warming_frames = []
        resources = runtime_resources.Resources('.stem/')


        self.get_logger().info('Initializing Completed.')

        # test_replay_buffer = learning_utils.ReplayBuffer(3, 100)
        # test_replay_buffer.append(0, [0]*10, [0]*128)
        # test_replay_buffer.append(1, [1]*10, [1]*128)
        # test_replay_buffer.append(0, [2]*10, [2]*128)
        # test_replay_buffer.append(2, [3]*10, [3]*128)

        # self.get_logger().info(str(test_replay_buffer.length()))

        # for index, frame, embedding in replay_buffer.popleft_each():
        #     print('T', index, frame, embedding)
        #     replay_buffer.append_back_buffer(index + 1, frame, embedding)

        # replay_buffer.swap_buffer()

        # for index, frame, embedding in replay_buffer.iterate():
        #     print(index, frame, embedding)

    def start(self):
        pass

    def test(self):
        pass
        # status = {'is_sensor_queue_full': False}
        # message = stem_utils.fill_message_from_dict(STEMStatus(), status)
        # self.status_publisher.publish(message)

    def on_receive_sensor_data(self, sensor_data):
        if len(sensor_data.segments) != self.sensor_data_segment_size:
            self.get_logger().warning('sensor_data segment size is incompatible.')
            return
        
        # print(sensor_data.segments)
        # time.sleep(2)
        self.sensor_data_queue.append(sensor_data.segments)
        self.status['sensor_data_queue_length'] = len(self.sensor_data_queue)
        if len(self.sensor_data_queue) == self.sensor_data_queue.maxlen:
            
            # print(a)
            # try:
            print(self.model(np.array([self.sensor_data_queue])))
            # except e:
            #     print('A: ')
            #     print(e)
            pass

        self.publish_status()

    def publish_status(self):
        self.status_publisher.publish(stem_utils.fill_message_from_dict(STEMStatus(), self.status))

    def on_receiver_supervise_signal(self, supervise_signal):
        self.get_logger().info(supervise_signal.supervised_state_name)
        pass


def main(args=None):
    rclpy.init(args=args)

    stem = STEM()
    stem.start()
    rclpy.spin(stem)

    stem.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()