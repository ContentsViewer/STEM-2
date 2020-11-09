import os
import pathlib
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from sklearn.cluster import KMeans
import pickle
from datetime import datetime
import queue

from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal
from stem_interfaces.msg import STEMStatus
from stem_interfaces.msg import Estimation
from stem_interfaces.srv import SaveModel

from stem_lib.stdlib.concurrent.thread import SingleThreadExecutor
from stem_lib.stdlib.stopwatch import Stopwatch
from stem_lib import utils as stem_utils
from stem_lib import learning_utils


class STEM(Node):
    def __init__(self):
        super().__init__('stem')

        self.get_logger().info('STEM is Awaken!')

        self.working_dir = stem_utils.load_parameter(self, 'working_dir', '.stem/' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        self.sensor_data_queue_size = stem_utils.load_parameter(self, 'sensor_data_queue_size', 100)
        self.state_names = stem_utils.load_parameter(self, 'state_names', ['touched', 'not_touched'])
        self.sensor_data_segment_size = stem_utils.load_parameter(self, 'sensor_data_segment_size', 2)
        self.sensor_sampling_rate_min = stem_utils.load_parameter(self, 'sensor_sampling_rate_min', 35)
        self.replay_buffer_maxlen = stem_utils.load_parameter(self, 'replay_buffer_maxlen', 100)
        self.nmin_samples_replay_buffer = stem_utils.load_parameter(self, 'nmin_samples_replay_buffer', 50)
        self.supervise_time_length_min = stem_utils.load_parameter(self, 'supervise_time_length_min', 0.5)
        self.self_learning_interval_sec = stem_utils.load_parameter(self, 'self_learning_interval_sec', 30)
        
        self.working_dir = pathlib.Path(self.working_dir)
        self.get_logger().info(f'Using "{self.working_dir}" as working directory.')

        try:
            learning_utils.restore_model(self.working_dir)

        except Exception as e:
            self.get_logger().warning(f'Failed to restore the model from {self.working_dir}. Create new model: {e}')
        

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

        self.save_model_service = self.create_service(SaveModel, 'save_model', self.on_request_save_model)
        self.status = {
            'sensor_data_queue_length': 0,
            'sensor_sampling_rate': 0,
            'is_estimating': False,
            'is_appending_initial_frames': False,
            'is_self_training': False,
            'is_supervised_training': False
        }

        self.sensor_data_sampleing_sw = Stopwatch()
        self.self_learning_sw = Stopwatch()
        self.supervise_time_length_sw = Stopwatch()
        self.current_supervised_state_name = 'none'
        self.supervise_signal_queue = queue.Queue(maxsize=1)

        self.sensor_data_queue = deque(maxlen=self.sensor_data_queue_size)
        self.model = learning_utils.make_model(
            self.sensor_data_queue_size, self.sensor_data_segment_size
        )

        self.replay_buffer = learning_utils.ReplayBuffer(len(self.state_names), self.replay_buffer_maxlen)
        self.initial_frames = []
        self.state_classifier = KMeans(n_clusters=len(self.state_names))
        self.state_name_id_bimapper = learning_utils.StateNameIdBiMapper()
        # self.compute_executor = SingleThreadExecutor()
        self.compute_executor = ThreadPoolExecutor(max_workers=1)

        self.sensor_sampling_rate_average = 0

        self.get_logger().info('Initializing Completed.')

    def start(self):
        self.sensor_data_sampleing_sw.start()
        self.self_learning_sw.start()


    def on_receive_sensor_data(self, sensor_data):
        if len(sensor_data.segments) != self.sensor_data_segment_size:
            self.get_logger().warning('sensor_data segment size is incompatible.')
            return
        
        try:
            self.status['sensor_sampling_rate'] = 1 / (self.sensor_data_sampleing_sw.elapsed + 1e-8)
            self.sensor_data_sampleing_sw.restart()
            
            self.sensor_sampling_rate_average = 0.5 * ( self.status['sensor_sampling_rate'] + self.sensor_sampling_rate_average)
            # self.get_logger().info(f'{self.sensor_sampling_rate_average}')
            if self.sensor_sampling_rate_average < self.sensor_sampling_rate_min:
                self.get_logger().warning(f'Low sensor sampling rate detected. Restart to pickup sensor data: {self.sensor_sampling_rate_average}/{self.sensor_sampling_rate_min}')
                self.sensor_data_queue.clear()

            self.sensor_data_queue.append(sensor_data.segments)
            self.status['sensor_data_queue_length'] = len(self.sensor_data_queue)


            if len(self.sensor_data_queue) == self.sensor_data_queue.maxlen:
                if self.replay_buffer.length() < self.nmin_samples_replay_buffer:
                    self.initial_frames.append(self.sensor_data_queue)
                    if len(self.initial_frames) >= self.nmin_samples_replay_buffer:
                        if (not self.status['is_appending_initial_frames'] and
                            not self.status['is_self_training']):
                            self.status['is_appending_initial_frames'] = True
                            future = self.compute_executor.submit(
                                learning_utils.append_initial_frames,
                                self.initial_frames,
                                self.replay_buffer,
                                self.model,
                                self.state_classifier
                            )
                            future.add_done_callback(self.on_appended_initial_frames)
                else:
                    is_get_supervise_signal = False
                    try:
                        supervised_time, supervised_state_name = self.supervise_signal_queue.get_nowait()
                        if time.time() - supervised_time < 0.1:
                            is_get_supervise_signal = True
                    except queue.Empty:
                        pass
                        
                    if (is_get_supervise_signal and
                        not self.status['is_appending_initial_frames'] and
                        not self.status['is_supervised_training']):
                        self.status['is_supervised_training'] = True
                        future = self.compute_executor.submit(
                            self.train_supervised,
                            self.sensor_data_queue,
                            supervised_state_name
                        )


                    elif (not self.status['is_estimating'] and
                        not self.status['is_appending_initial_frames'] and
                        not self.status['is_self_training']):
                        self.status['is_estimating'] = True
                        future = self.compute_executor.submit(
                            learning_utils.estimate_state,
                            self.sensor_data_queue,
                            self.model,
                            self.state_classifier
                        )
                        future.add_done_callback(self.on_estimated)

        except Exception as e:
            self.get_logger().error(f'{e}')

        finally:
            self.publish_status()

    def on_appended_initial_frames(self, future):
        self.status['is_appending_initial_frames'] = False
        self.initial_frames.clear()

        self.publish_status()
    
    def on_estimated(self, future):
        try:
            state_id, frame, embedding = future.result()

            self.publish_estimation(state_id, str(self.state_name_id_bimapper.get_name(state_id)))

            if self.self_learning_sw.elapsed > self.self_learning_interval_sec:
                if (not self.status['is_self_training'] and
                    not self.status['is_supervised_training']):
                    self.status['is_self_training'] = True
                    future = self.compute_executor.submit(
                        learning_utils.train_model,
                        self.model,
                        self.replay_buffer,
                        frame,
                        embedding,
                        state_id
                    )
                    future.add_done_callback(self.on_trained)
                self.self_learning_sw.restart()

        except Exception as e:
            self.get_logger().error(f'{e}')

        finally:
            self.status['is_estimating'] = False
            self.publish_status()

        

    def on_trained(self, future):
        try:
            anchor_frame, anchor_embedding, anchor_state_id = future.result()
            learning_utils.recal_embeddings(self.model, self.replay_buffer)
            self.replay_buffer.append(anchor_state_id, anchor_frame, anchor_embedding)

        except Exception as e:
            self.get_logger().error(f'{e}')

        finally:
            self.status['is_self_training'] = False
            self.publish_status()

    def train_supervised(self, frame, supervised_state_name):
        try:
            [embedding], _ = self.model(np.array([frame]))
            [estimated_state_id] = state_classifier.predict([embedding])
            supervised_state_id = self.state_name_id_bimapper.get_id(supervised_state_name)
            if supervised_state_id is None:
                self.state_name_id_bimapper.bind(supervised_state_name, estimated_state_id)
                supervised_state_id = estimated_state_id
            learning_utils.train_model(self.model, self.replay_buffer, frame, embedding, supervised_state_id)
            learning_utils.recal_embeddings(self.model, self.replay_buffer)
            self.replay_buffer.append(supervised_state_id, frame, embedding)

        except Exception as e:
            self.get_logger().error(f'{e}')

        finally:
            self.status['is_supervised_training'] = False
            self.publish_status()


    def publish_estimation(self, state_id, state_name):
        estimation = Estimation()
        estimation.state_name = state_name
        estimation.state_id = state_id
        self.estimation_publisher.publish(estimation)

    def publish_status(self):
        self.status_publisher.publish(stem_utils.fill_message_from_dict(STEMStatus(), self.status))

    def on_receive_supervise_signal(self, supervise_signal):
        # self.get_logger().info(supervise_signal.supervised_state_name)
        if self.current_supervised_state_name != supervise_signal.supervised_state_name:
            self.current_supervised_state_name = supervise_signal.supervised_state_name
            self.supervise_time_length_sw.restart()
        
        if self.supervise_time_length_sw.elapsed > self.supervise_time_length_min:
            self.supervise_time_length_sw.restart()
            if self.current_supervised_state_name != 'none':
                try:
                    self.supervise_signal_queue.put_nowait(
                        (time.time(), self.current_supervised_state_name)
                    )
                except queue.Full:
                    pass

    def on_request_save_model(self, request, response):
        try:
            learning_utils.save_model(self.working_dir, self.replay_buffer, self.model)
        except Exception as e:
            response.success = False
            self.get_logger().error(f'Failed to save model: {e}')
        else:
            response.success = True
        
        return response

def main(args=None):
    rclpy.init(args=args)

    stem = STEM()
    stem.start()
    rclpy.spin(stem)

    stem.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()