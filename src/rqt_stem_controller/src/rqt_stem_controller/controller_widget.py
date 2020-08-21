import os

from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtWidgets import QPushButton

from qt_gui.ros_package_helper import get_package_path

class ControllerWidget(QWidget):
    def __init__(self, parent=None):
        super(ControllerWidget, self).__init__(parent)

        package_path = get_package_path('rqt_stem_controller')
        ui_file = os.path.join(package_path, 'share', 'rqt_stem_controller', 'resource', 'STEMController.ui')
        loadUi(ui_file, self)

        # btn1 = QPushButton("Button 1", self)
        # btn1.move(30, 50)
        # btn1.clicked.connect(self.buttonClicked)  

    def keyPressEvent(self, e):

        # # エスケープキーを押すと画面が閉じる
        # if e.key() == Qt.Key_Escape:
        #     self._widget.shutdown()
        
        print(e.key())
        self.supervised_state_name = e.key()
        
    def keyReleaseEvent(self, e):
        print('RELEAASS')
        self.supervised_state_name = 'RELEAASE'

    def buttonClicked(self):
        print('OK')
