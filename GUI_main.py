from PyQt6 import QtWidgets, QtCore
import sys
from GUI_TabGaussJordan import *
from GUI_TabAX_B import *
from GUI_TabDependencia import *
from GUI_TabVectores import *
from GUI_Matrices import *
from Tab_Inversa import *
from GUI_TabDetermianante import *
from GUI_TabMetodosNumericos import *
from widgets import *


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Álgebra Lineal ")
        self.resize(1200, 820)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(TabGaussJordan(), "Gauss-Jordan (Ab)")
        tabs.addTab(TabAXeqB(), "AX=B")
        tabs.addTab(TabProg4(), "Dependencia")         
        tabs.addTab(TabVectores(), "Vectores")
        tabs.addTab(TabProg5(), "Operaciones con Matrices")
        tabs.addTab(TabProg6(), "Inversa")
        tabs.addTab(TabDeterminante(), "Determinante")   
        tabs.addTab(TabMetodosNumericos(), "Métodos Numéricos") 

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Listo")

        # Oculta las flechas de TODOS los SpinBox del árbol y desactiva la rueda
        hide_all_spin_buttons(self)

# =========================
#   main() con tema claro
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_green_theme(app)  

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()