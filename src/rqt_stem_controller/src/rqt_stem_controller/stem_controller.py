
from python_qt_binding.QtWidgets import QVBoxLayout, QWidget
from rqt_gui_py.plugin import Plugin

class STEMController(Plugin):
    def __init__(self, context):
        super(STEMController, self).__init__(context)

        self.setObjectName('STEMController')

        self._context = context

        self._widget = QWidget()
        self._context.add_widget(self._widget)
    
    
    def shutdown_plugin(self):
        pass