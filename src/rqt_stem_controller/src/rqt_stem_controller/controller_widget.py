import os

from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtWidgets import QPushButton
from python_qt_binding.QtWidgets import QListWidgetItem
from python_qt_binding.QtGui import QIcon

from qt_gui.ros_package_helper import get_package_path

class SuperviseButton(QPushButton):
    def __init__(self, state_name):
        super(SuperviseButton, self).__init__(state_name)

        self._state_name = state_name
        self._is_pressed = False

        self.pressed.connect(self._on_pressed)
        self.released.connect(self._on_released)
        
    
    @property
    def state_name(self):
        return self._state_name

    @property
    def is_pressed(self):
        return self._is_pressed
    
    def _on_pressed(self):
        self._is_pressed = True
    
    def _on_released(self):
        self._is_pressed = False
    
    
class ControllerWidget(QWidget):
    def __init__(self, parent=None):
        super(ControllerWidget, self).__init__(parent)

        package_path = get_package_path('rqt_stem_controller')
        ui_file = os.path.join(package_path, 'share', 'rqt_stem_controller', 'resource', 'STEMController.ui')
        loadUi(ui_file, self)

        self.supervise_buttons = []
        # self.supervise_button_layout.addWidget(SuperviseButton('TEST'))
        # self.supervise_button_layout.addWidget(SuperviseButton('TEST2'))
        # self.supervise_button_layout.addWidget(SuperviseButton('TEST3'))

        # self.state_name_combo_box.setEditable(True)
        # self.state_name_remove_button.setIcon(QIcon.fromTheme('list-remove'))
        # self.state_name_remove_button.clicked.connect(self.remove_current_state)

    def set_supervise_buttons(self, state_names):
        for state_name in state_names:
            button = SuperviseButton(state_name)
            self.supervise_buttons.append(button)
            self.supervise_button_layout.addWidget(button)

    def remove_current_state(self):
        current_index = self.state_name_combo_box.currentIndex()
        self.state_name_combo_box.removeItem(current_index)
        self.state_name_combo_box.clearEditText()