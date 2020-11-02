from launch import LaunchDescription
from launch_ros.actions import Node
 
def generate_launch_description():
    return LaunchDescription([
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
            package='stem',
            executable='stem',
            parameters=[{
                'sensor_data_queue_size': 100,
                'state_names': ["touched", "not_touched"],
                'sensor_data_segment_size': 2,
                'replay_buffer_maxlen': 100,
                'nmin_samples_replay_buffer': 50,
                'sensor_sampling_rate_min': 35,
                'working_dir': '.stem/balloon_contact'
            }],
            # emulate_tty=True,
            output='screen'
        ),
    ])
