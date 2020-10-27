
from python_qt_binding.QtWidgets import QVBoxLayout, QWidget
from python_qt_binding.QtCore import Slot, QSignalMapper, QTimer, qWarning
from rqt_gui_py.plugin import Plugin
from .controller_widget import ControllerWidget
from .controller_widget import SuperviseButton

from rclpy.qos import QoSProfile
from ros2param.api import call_get_parameters
from rcl_interfaces.msg import ParameterType

from stem_interfaces.msg import SuperviseSignal

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

        self._publisher = self._node.create_publisher(
            SuperviseSignal, 
            'supervise_signal', 
            qos_profile=QoSProfile(depth=10)
        )
        # self._timer = self._node.create_timer(0.02, self.update)
        
        try:
            response = call_get_parameters(node=self._node, node_name='stem', parameter_names=['state_names'])
            pvalue = response.values[0]
            if pvalue.type != ParameterType.PARAMETER_STRING_ARRAY:
                raise RuntimeError('type of parameter "state_names" is Not STRING_ARRAY')

            self.state_names = pvalue.string_array_value

        except RuntimeError as e:
            self._node.get_logger().warn(f'Failed to get parameter "stem/state_names". Please reload the plugin: {e}')

        self._widget.set_supervise_buttons(self.state_names)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(20)
        # print('ok')

    def update(self):
        supervised_state_name = 'none'
        for supervise_button in self._widget.supervise_buttons:
            if supervise_button.is_pressed:
                supervised_state_name = supervise_button.state_name
                break
        # print('test')
        supervise_signal = SuperviseSignal()
        supervise_signal.supervised_state_name = supervised_state_name
        self._publisher.publish(supervise_signal)

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

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass

    def shutdown_plugin(self):
        pass