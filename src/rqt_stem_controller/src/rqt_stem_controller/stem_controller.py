
from python_qt_binding.QtWidgets import QVBoxLayout, QWidget
from rqt_gui_py.plugin import Plugin
from .controller_widget import ControllerWidget

from rclpy.qos import QoSProfile
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

        self._publisher = self._node.create_publisher(SuperviseSignal, 'supervise_signal', qos_profile=QoSProfile(depth=10))
        self._timer = self._node.create_timer(0.02, self.timer_callback)

    
    def timer_callback(self):

        if self._widget.is_pressed_supervise_button:
            supervised_state_name = self._widget.state_name_combo_box.currentText()
            if supervised_state_name == '':
                return

            supervise_signal = SuperviseSignal()
            supervise_signal.supervised_state_name = supervised_state_name
            self._publisher.publish(supervise_signal)
            self._widget.supervised_state_label.setText(supervised_state_name)
        else:
            self._widget.supervised_state_label.setText('NONE')

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass

    def shutdown_plugin(self):
        pass