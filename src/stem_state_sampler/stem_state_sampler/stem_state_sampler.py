

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from collections import deque

from stem_interfaces.msg import STEMStatus
from stem_interfaces.msg import GeneralSensorData
from stem_interfaces.msg import SuperviseSignal
from stem_interfaces.srv import SaveSamples


from stem_lib.stdlib.stopwatch import Stopwatch
from stem_lib import utils as stem_utils
from stem_lib import learning_utils



class StemStateSampler(Node):
    def __init__(self):
        super().__init__('stem_state_sampler')

        self.get_logger().info('Awaken.')

        self.working_dir = stem_utils.load_parameter(self, 'working_dir', '.stem/samples')
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

        self.status_publisher = self.create_publisher(
            STEMStatus,
            'stem_status',
            QoSProfile(depth=10)
        )

        self.save_samples_service = self.create_service(SaveSamples, 'save_samples', self.on_request_save_samples)

        self.status = {
            'sensor_data_queue_length': 0,
            'sensor_sampling_rate': 0,
            'is_estimating': False,
            'is_appending_initial_frames': False,
            'is_self_training': False,
            'is_supervised_training': False
        }

        self.sensor_sampleing_watchdog = stem_utils.SamplingRateWatchdog(self.sensor_sampling_rate_min)

        self.sensor_data_queue = deque(maxlen=self.sensor_data_queue_size)

        self.supervise_state_listener = stem_utils.StateChangeListener()

        self.get_logger().info('Ready.')

    def start(self):
        self.sensor_sampleing_watchdog.start()

    def on_receive_sensor_data(self, sensor_data):
        if not self.sensor_sampleing_watchdog.lap():
            self.get_logger().warning(
                f'Low sensor sampling rate detected. Restart to pickup sensor data: {self.sensor_sampleing_watchdog.sampling_rate}/{self.sensor_sampling_rate_min}')

            self.sensor_data_queue.clear()
    
        self.status['sensor_sampling_rate'] = self.sensor_sampleing_watchdog.sampling_rate

        changed, state_name = self.supervise_state_listener.has_changed(timeout=0.2)
        if changed:
            self.get_logger().info(str(state_name))

        self.publish_status()


    def on_receive_supervise_signal(self, supervise_signal):
        if supervise_signal.supervise_state_name == 'none-supervised':
            self.supervise_state_listener.update(None)
            return
        self.supervise_state_listener(supervise_signal.supervise_state_name)
    
    def publish_status(self):
        self.status_publisher.publish(stem_utils.fill_message_from_dict(STEMStatus(), self.status))


    def on_request_save_samples(self, request, response):
        # try:
        #     learning_utils.save_model(self.working_dir, self.replay_buffer, self.model, self.state_classifier)
        # except Exception as e:
        #     response.success = False
        #     self.get_logger().error(f'Failed to save model: {e}')
        # else:
        #     response.success = True
        response.success = True
        self.get_logger().info('Saving samples...')
        return response

def main(args=None):
    rclpy.init(args=args)

    stem_state_sampler = StemStateSampler()
    stem_state_sampler.start()
    rclpy.spin(stem_state_sampler)

    stem_state_sampler.destroy_node()
    stem_state_sampler.shutdown()

if __name__ == '__main__':
    main()