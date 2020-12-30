from launch import LaunchDescription
from launch_ros.actions import Node
 
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stem_parameter_server',
            executable='stem_parameter_server',
            parameters=[{
                'state_names': ["inflating", "shrinking", "baunded", "non"],
                'sensor_sampling_rate_min': 35,
            }],
        ),
        Node(
            package='sensor_driver',
            executable='balloon_sensor_driver',
            parameters=[{
                'port': '/dev/ttyACM0',
                'boudrate': 19200,
                'read_period_sec': 0.02
            }],
        ),
        Node(
            package='stem_state_sampler',
            executable='stem_state_sampler',
            parameters=[{
                'state_names': ["inflating", "shrinking", "baunded", "non"],
                'sensor_sampling_rate_min': 35,
                'sensor_data_queue_size': 100,
                'sensor_data_segment_size': 2,
                # 'working_dir': '.stem/samples',
            }],
            # emulate_tty=True,
            output='screen'
        ),
    ])
