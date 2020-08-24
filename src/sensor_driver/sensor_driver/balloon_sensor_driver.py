import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from stem_interfaces.msg import BalloonSensorData
from stem_interfaces.msg import GeneralSensorData

from stem_lib import sensor_utils
from stem_lib import utils as stem_utils

import sys


class BalloonSensorDriver(Node):

    def __init__(self):
        super().__init__('balloon_sensor_driver')

        self.declare_parameter("port", '/dev/ttyACM0')
        self.declare_parameter("baudrate", 19200)
        self.declare_parameter("read_period_sec", 0.02)

        port = self.get_parameter('port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        read_period_sec = self.get_parameter('read_period_sec').get_parameter_value().double_value
        
        # self.publisher_ = self.create_publisher(BalloonSensorData, 'balloon_sensor_data', QoSPresetProfiles.SENSOR_DATA.value)
        self.balloon_sensor_data_publisher = self.create_publisher(BalloonSensorData, 'balloon_sensor_data', 10)
        self.general_sensor_data_publisher = self.create_publisher(GeneralSensorData, 'general_sensor_data', 10)
        self.timer = self.create_timer(read_period_sec, self.timer_callback)
        self.ser = sensor_utils.begin_serial(port, baudrate)

    def timer_callback(self):
        segments = sensor_utils.read_latest_line(self.ser).decode().split()

        if len(segments) != 3:
            self.get_logger().warning('not supported sensor. should have 3 segments.')
            return

        balloon_sensor_data = BalloonSensorData()
        balloon_sensor_data.timestamp = int(segments[0])
        balloon_sensor_data.pulse_width = int(segments[1])
        balloon_sensor_data.flow_amount = int(segments[2])
        self.balloon_sensor_data_publisher.publish(balloon_sensor_data)

        general_sensor_data = GeneralSensorData()
        general_sensor_data.segments = [ float(segment) for segment in segments] 
        self.general_sensor_data_publisher.publish(general_sensor_data)


def main(args=None):
    rclpy.init(args=args)

    driver= BalloonSensorDriver()
    rclpy.spin(driver)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    driver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()