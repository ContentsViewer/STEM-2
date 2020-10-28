import os
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor

from sklearn.cluster import KMeans
import pickle

from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal
from stem_interfaces.msg import STEMStatus
from stem_interfaces.msg import Estimation

from stem_lib.stdlib import runtime_resources
from stem_lib.stdlib.concurrent.thread import SingleThreadExecutor
from stem_lib.stdlib.stopwatch import Stopwatch
from stem_lib import utils as stem_utils
from stem_lib import learning_utils


class STEM(Node):
    def __init__(self):
        super().__init__('stem')

        self.get_logger().info('STEM is Awaken!')

        self.sensor_data_queue_size = stem_utils.load_parameter(self, 'sensor_data_queue_size', 100)
        self.state_names = stem_utils.load_parameter(self, 'state_names', ['touched', 'not_touched'])
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
            self.on_receive_supervise_signal,
            QoSProfile(depth=10)
        )

        self.status_publisher = self.create_publisher(
            STEMStatus,
            'stem_status',
            QoSProfile(depth=10)
        )

        self.estimation_publisher = self.create_publisher(
            Estimation,
            'estimation',
            QoSProfile(depth=10)
        )

        self.status = {
            'sensor_data_queue_length': 0,
            'sensor_sampling_rate': 0
        }

        self.sensor_data_sampleing_sw = Stopwatch()


        self.sensor_data_queue = deque(maxlen=self.sensor_data_queue_size)
        self.model = learning_utils.make_model(
            self.sensor_data_queue_size, self.sensor_data_segment_size
        )

        self.replay_buffer = learning_utils.ReplayBuffer(len(self.state_names), self.replay_buffer_maxlen)
        self.initial_frames = []
        self.state_classifier = KMeans(n_clusters=len(self.state_names))
        self.state_name_id_bimapper = learning_utils.StateNameIdBiMapper()
        self.compute_executor = SingleThreadExecutor()

        resources = runtime_resources.Resources('.stem/')


        # self.timer = self.create_timer(0.1, self.test)
        # self.test_thread_worker = ThreadWorker()

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
        # X = np.array([[1, 2], [1, 4], [1, 0],
        #            [10, 2], [10, 4], [10, 0]])
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        # self.get_logger().info(str(kmeans.labels_))
        # kmeans.predict([[0, 0], [12, 3]])
        # self.get_logger().info(str(kmeans.cluster_centers_))

        # pkl_filename = "pickle_model.pkl"
        # with open(pkl_filename, 'wb') as file:
        #     pickle.dump(kmeans, file)

        # with open(pkl_filename, 'rb') as file:
        #     pickle_model = pickle.load(file)
        
        # self.get_logger().info(str(pickle_model.labels_))
        # pickle_model.predict([[0, 0], [12, 3]])
        # self.get_logger().info(str(pickle_model.cluster_centers_))


    def start(self):
        self.sensor_data_sampleing_sw.start()

    def reset(self):
        pass

    def test(self, mes):
        time.sleep(2)

        self.get_logger().info(str(mes))

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

        self.status['sensor_sampling_rate'] = 1 / (self.sensor_data_sampleing_sw.elapsed + 1e-8)
        self.sensor_data_sampleing_sw.restart()

        if len(self.sensor_data_queue) == self.sensor_data_queue.maxlen:
            if self.replay_buffer.length() < self.nmin_samples_replay_buffer:
                self.initial_frames.append(self.sensor_data_queue)
                if len(self.initial_frames) >= self.nmin_samples_replay_buffer:
                    self.compute_executor.run(
                        learning_utils.append_initial_frames, 
                        args=(self.initial_frames, self.replay_buffer, self.model, self.state_classifier),
                        done_callback=lambda exit_status: self.initial_frames.clear()
                    )
                    
            else:
                self.compute_executor.run(
                    learning_utils.estimate_state,
                    args=(self.sensor_data_queue, self.model, self.state_classifier),
                    done_callback=self.on_estimated
                )

        self.publish_status()

    
    def on_estimated(self, exit_status):
        try:
            state_id, frame, embedding = exit_status.result()
        except BaseException as exc:
            self.get_logger().error(str(exc))
            return

        self.publish_estimation(state_id, str(self.state_name_id_bimapper.get_name(state_id)))
        

    def publish_estimation(self, state_id, state_name):
        estimation = Estimation()
        estimation.state_name = state_name
        estimation.state_id = state_id
        self.estimation_publisher.publish(estimation)

    def publish_status(self):
        self.status_publisher.publish(stem_utils.fill_message_from_dict(STEMStatus(), self.status))

    def on_receive_supervise_signal(self, supervise_signal):
        # self.get_logger().info(supervise_signal.supervised_state_name)
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