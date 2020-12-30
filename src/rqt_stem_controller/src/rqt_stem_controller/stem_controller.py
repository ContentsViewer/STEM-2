import time
import queue

from python_qt_binding.QtWidgets import QVBoxLayout, QWidget
from python_qt_binding.QtCore import Slot, QSignalMapper, QTimer, qWarning
from rqt_gui_py.plugin import Plugin


from rclpy.qos import QoSProfile
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

from .controller_widget import ControllerWidget
from .controller_widget import SuperviseButton
from .controller_widget import SignalLampController

from stem_lib.stdlib.concurrent.thread import run_once_async
from stem_lib import ros_utils

from stem_interfaces.msg import SuperviseSignal
from stem_interfaces.msg import Estimation
from stem_interfaces.msg import STEMStatus
from stem_interfaces.srv import SaveModel


class STEMController(Plugin):
    def __init__(self, context):
        super(STEMController, self).__init__(context)

        self.setObjectName('STEMController')

        self._node = context.node
        self._context = context

        self._widget = ControllerWidget()
        if context.serial_number() > 1:
            self._widget.setWindowTitle(
                self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        
        self._context.add_widget(self._widget)

        self._supervise_signal_publisher = self._node.create_publisher(
            SuperviseSignal, 
            'supervise_signal', 
            qos_profile=QoSProfile(depth=10)
        )

        self._estimation_receiver = self._node.create_subscription(
            Estimation,
            'estimation',
            self.on_receive_estimation,
            QoSProfile(depth=10)
        )

        self._stem_status_receiver = self._node.create_subscription(
            STEMStatus,
            'stem_status',
            self.on_receive_stem_status,
            QoSProfile(depth=10)
        )

        self._save_model_client = self._node.create_client(SaveModel, 'save_model')
        # self._timer = self._node.create_timer(0.02, self.update)
    

        self._stem_get_parameters_client = self._node.create_client(GetParameters, 'stem_parameter_server/get_parameters')
        self.stem_params = {
            'state_names': None,
            'sensor_sampling_rate_min': None
        }

        self.command_queue = queue.Queue()
        # self.fetch_set_state_names()
        # run_once_async(self.fetch_state_names)
        self.fetch_stem_parameters()
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(20)

        self._widget.save_model_button.clicked.connect(self.request_save_model)

        self._lamp_controller = SignalLampController()
        self._busy_rate = 0.0
        # print('ok')

    def fetch_stem_parameters(self):
        def on_receive_response(future):
            try:
                response = future.result()

                state_names, sensor_sampling_rate_min = response.values

                try:
                    self.stem_params['state_names'] = ros_utils.get_parameter_value(
                        state_names, 
                        [ParameterType.PARAMETER_STRING_ARRAY]
                    )
                    self.command_queue.put(
                        (
                            self._widget.set_supervise_buttons, 
                            (self.stem_params['state_names'],), 
                            {}
                        )
                    )
                except Exception as e:
                    self._node.get_logger().warn(f'Failed to get parameter "stem/state_names": {e}')
                
                try:
                    self.stem_params['sensor_sampling_rate_min'] = ros_utils.get_parameter_value(
                        sensor_sampling_rate_min, 
                        [ParameterType.PARAMETER_INTEGER, ParameterType.PARAMETER_DOUBLE]
                    )
                except Exception as e:
                    self._node.get_logger().warn(f'Failed to get parameter "stem/sensor_sampling_rate_min": {e}')
                        

            except Exception as e:
                self._node.get_logger().warn(f'Failed to get stem parameters. Please reload the plugin: {e}')

        try:
            future = ros_utils.request_get_parameters_async(
                self._stem_get_parameters_client,
                parameter_names=['state_names', 'sensor_sampling_rate_min']
            )
        except Exception as e:
            self._node.get_logger().error(f'Failed to request stem/get_parameters service: {e}')
            return

        future.add_done_callback(on_receive_response)
            

    def update(self):
        try:
            command = self.command_queue.get_nowait()
            command[0](*command[1], **command[2])
        except queue.Empty:
            pass

        supervise_state_name = 'none-supervised'
        for supervise_button in self._widget.supervise_buttons:
            if supervise_button.is_pressed:
                supervise_state_name = supervise_button.state_name
                break
        # print('test')
        supervise_signal = SuperviseSignal()
        supervise_signal.supervise_state_name = supervise_state_name
        self._supervise_signal_publisher.publish(supervise_signal)


        self._widget.process_busy_rate.setValue(self._busy_rate)

        self._lamp_controller.update()
        # print(self._widget.test_button.is_pressed)
        
        # if self._widget.is_pressed_supervise_button:
        #     supervised_state_name = self._widget.state_name_combo_box.currentText()
        #     if supervised_state_name == '':
        #         return

        #     supervise_signal = SuperviseSignal()
        #     supervise_signal.supervised_state_name = supervised_state_name
        #     self._publisher.publish(supervise_signal)
        #     self._widget.supervised_state_label.setText(supervised_state_name)
        # else:
        #     self._widget.supervised_state_label.setText('NONE')

    def on_receive_estimation(self, estimation):
        self._widget.estimation_state_name.setText(estimation.state_name)
        self._widget.estimation_state_id.display(estimation.state_id)

        self._lamp_controller.trigger(self._widget.estimation_signal_lamp)

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass

    def shutdown_plugin(self):
        self._timer.stop()
        self._node.destroy_subscription(self._estimation_receiver)
        self._node.destroy_publisher(self._supervise_signal_publisher)
        # pass
    
    def request_save_model(self):
        try:
            future = ros_utils.request_service_async(self._save_model_client, SaveModel.Request())
        except Exception as e:
            self._node.get_logger().error(f'Failed to request save_model service: {e}')
            return

        def on_receive_response(future):
            try:
                response = future.result()
                # self._node.get_logger().info(str(response.success))
                self._lamp_controller.trigger(self._widget.save_model_lamp)

            except Exception as e:
                self._node.get_logger().error(f'Exception has been raised while requesting for save model service: {e}')
            
            self._widget.save_model_button.setEnabled(True)


        future.add_done_callback(on_receive_response)
        self._widget.save_model_button.setEnabled(False)
    
    def on_receive_stem_status(self, status):
            
        min_rate = self.stem_params['sensor_sampling_rate_min']
        if min_rate is not None:
            current_rate = status.sensor_sampling_rate
            self._busy_rate = max(100 - max(current_rate - min_rate, 0), 0)
            # self._node.get_logger().info(f'{busy_rate}')
        