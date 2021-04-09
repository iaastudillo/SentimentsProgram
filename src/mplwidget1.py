from PyQt5.QtWidgets import*

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure
import matplotlib

import numpy as np
import seaborn as sns
    
class MplWidget1(QWidget):
    
    def __init__(self, parent = None):

        QWidget.__init__(self, parent)
        
        self.canvas_1 = FigureCanvas(Figure())
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas_1)
        self.canvas_1.matriz = self.canvas_1.figure.add_subplot(111)
        self.setLayout(vertical_layout)