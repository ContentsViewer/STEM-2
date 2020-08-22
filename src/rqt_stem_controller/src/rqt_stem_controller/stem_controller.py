
from python_qt_binding.QtWidgets import QVBoxLayout, QWidget
from rqt_gui_py.plugin import Plugin
from .controller_widget import ControllerWidget

from rclpy.qos import QoSProfile
from std_msgs.msg import String

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

        self._publisher = self._node.create_publisher(String, 'supervised_state_name', qos_profile=QoSProfile(depth=10))
        self._timer = self._node.create_timer(1.0, self.timer_callback)

        self.state_list = ['not_touched', 'touched']

        for state in self.state_list:
            pass
    
    def timer_callback(self):
        print('test')

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass

    def shutdown_plugin(self):
        pass