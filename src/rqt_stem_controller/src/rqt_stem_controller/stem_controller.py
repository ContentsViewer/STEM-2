
from python_qt_binding.QtWidgets import QVBoxLayout, QWidget
from rqt_gui_py.plugin import Plugin
from .controller_widget import ControllerWidget

class STEMController(Plugin):
    def __init__(self, context):
        super(STEMController, self).__init__(context)

        self.setObjectName('STEMController')

        self._context = context

        self._widget = ControllerWidget()
        if context.serial_number() > 1:
            self._widget.setWindowTitle(
                self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        

        self._context.add_widget(self._widget)
    

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass

    def shutdown_plugin(self):
        pass