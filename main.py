import sys
from PyQt5 import QtWidgets

from src import Sentiment_menu

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Sentiment_menu.MyApp()
    window.show()
    sys.exit(app.exec_())
