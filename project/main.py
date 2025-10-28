from PySide6 import QtWidgets
from app.main_window import MainWindow
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
