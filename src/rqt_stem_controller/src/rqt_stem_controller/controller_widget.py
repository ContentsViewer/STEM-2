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
        self.state2widgets_mapper = {}
        self.add_state('item')
        self.add_state('item')

    def supervise_button_pressed(self):
        self.is_pressed_supervise_button = True

    def supervise_button_released(self):
        self.is_pressed_supervise_button = False

    def add_state(self, state_name):
        if state_name in self.state2widgets_mapper:
            return
        
        item = QListWidgetItem(state_name)
        self.state2widgets_mapper[state_name] = item
        self.state_list_widget.addItem(item)
    
    def remove_state(self, state_name):
        if not state_name in self.state2widgets_mapper:
            return
        
        self.state_list_widget.removeItemWidget(self.state2widgets_mapper[state_name])
        self.state2widgets_mapper.pop(state_name)

    # def keyPressEvent(self, e):

    #     # # エスケープキーを押すと画面が閉じる
    #     # if e.key() == Qt.Key_Escape:
    #     #     self._widget.shutdown()
        
    #     print(e.key())
    #     self.supervised_state_name = e.key()
        
    # def keyReleaseEvent(self, e):
    #     print('RELEAASS')
    #     self.supervised_state_name = 'RELEAASE'
