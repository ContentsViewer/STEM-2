
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from stem_lib import utils as stem_utils

class StemParameterServer(Node):
    def __init__(self):
        super().__init__('stem_parameter_server')
        
        self.sensor_sampling_rate_min = stem_utils.load_parameter(self, 'sensor_sampling_rate_min', 35)
        self.state_names = stem_utils.load_parameter(self, 'state_names', ['touched', 'not_touched'])

        self.get_logger().info('Awaken.')

def main(args=None):
    rclpy.init(args=args)

    stem_parameter_server = StemParameterServer()
    rclpy.spin(stem_parameter_server)

    stem_parameter_server.destroy_node()
    stem_parameter_server.shutdown()

if __name__ == '__main__':
    main()