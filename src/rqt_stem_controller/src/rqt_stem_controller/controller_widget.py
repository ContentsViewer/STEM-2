import os

from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtWidgets import QPushButton
from python_qt_binding.QtWidgets import QListWidgetItem
from python_qt_binding.QtGui import QIcon

from qt_gui.ros_package_helper import get_package_path

class ControllerWidget(QWidget):
    def __init__(self, parent=None):
        super(ControllerWidget, self).__init__(parent)

        package_path = get_package_path('rqt_stem_controller')
        ui_file = os.path.join(package_path, 'share', 'rqt_stem_controller', 'resource', 'STEMController.ui')
        loadUi(ui_file, self)

        self.is_pressed_supervise_button = False
        self.supervise_button.pressed.connect(self.supervise_button_pressed)
        self.supervise_button.released.connect(self.supervise_button_released)

        self.state_name_combo_box.setEditable(True)
        self.state_name_remove_button.setIcon(QIcon.fromTheme('list-remove'))
        self.state_name_remove_button.clicked.connect(self.remove_current_state)

    def supervise_button_pressed(self):
        self.is_pressed_supervise_button = True

    def supervise_button_released(self):
        self.is_pressed_supervise_button = False

    def remove_current_state(self):
        current_index = self.state_name_combo_box.currentIndex()
        self.state_name_combo_box.removeItem(current_index)
        self.state_name_combo_box.clearEditText()