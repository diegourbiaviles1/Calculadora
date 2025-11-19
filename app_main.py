# app_gui.py (Versión Modularizada Final)

from __future__ import annotations
import sys
from PyQt6 import QtWidgets, QtCore, QtGui

from utilidad import *
from sistema_lineal import *
from homogeneo import *
from algebra_vector import *
from matrices import *
from inversa import *
from determinante import *
from metodos_numericos import *
from notacion_posicional import *

from widgets import hide_all_spin_buttons

from GUI_TabGaussJordan import TabGaussJordan
from GUI_TabAX_B import TabAXeqB
from GUI_TabDependencia import TabProg4  # Asumiendo que esta es la clase para Dependencia
from GUI_TabVectores import TabVectores
from GUI_TabMatrices import TabProg5      # Asumiendo que esta es la clase para Operaciones con Matrices
from GUI_TabInversa import TabProg6       # Asumiendo que esta es la clase para Inversa
from GUI_TabDeterminante import TabDeterminante
from GUI_TabMetodosNumericos import TabMetodosNumericos


# =========================
# 3. ESTILO Y TEMA (Contenido que aplica a toda la aplicación)
# =========================
def apply_green_theme(app: QtWidgets.QApplication):
    """
    Tema verde moderno: botones con hover/focus, tabs marcados en verde,
    inputs y tablas con acentos verdes y esquinas redondeadas.
    """
    # ---- Paleta base clara con Highlights en verde ----
    palette = QtGui.QPalette()
    bg        = QtGui.QColor(246, 248, 246)
    panel     = QtGui.QColor(255, 255, 255)
    text      = QtGui.QColor(28, 28, 28)
    subtext   = QtGui.QColor(95, 95, 95)
    green     = QtGui.QColor(34, 139, 34)
    greenDark = QtGui.QColor(22, 115, 22)
    greenLite = QtGui.QColor(227, 245, 229)

    palette.setColor(QtGui.QPalette.ColorRole.Window, bg)
    palette.setColor(QtGui.QPalette.ColorRole.Base, panel)
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(240, 243, 240))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Text, text)
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, subtext)
    palette.setColor(QtGui.QPalette.ColorRole.Button, panel)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, green)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(255, 255, 235))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
    app.setPalette(palette)
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))

    # ---- Hoja de estilos (QSS) ----
    app.setStyleSheet(f"""
        /* Tipografía y fondos */
        QWidget {{
            font-size: 10.5pt;
            color: rgb({text.red()},{text.green()},{text.blue()});
            background-color: rgb({bg.red()},{bg.green()},{bg.blue()});
        }}

        QGroupBox {{
            border: 1px solid rgba(0,0,0,20%);
            border-radius: 10px;
            margin-top: 12px;
            background: rgb({panel.red()},{panel.green()},{panel.blue()});
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 4px 8px;
            color: rgb({green.darker(120).red()},{green.darker(120).green()},{green.darker(120).blue()});
            font-weight: 600;
        }}

        /* Botones */
        QPushButton {{
            border: 1px solid rgba(0,0,0,12%);
            padding: 8px 14px;
            border-radius: 10px;
            background: rgb({panel.red()},{panel.green()},{panel.blue()});
        }}
        QPushButton:disabled {{
            color: rgba(0,0,0,45%);
            border-color: rgba(0,0,0,8%);
            background: rgba(0,0,0,3%);
        }}

        /* Primary (VERDE) */
        QPushButton#btn-primary {{
            background: rgb({green.red()},{green.green()},{green.blue()});
            border: 1px solid rgba(0,0,0,10%);
            color: white;
            font-weight: 600;
        }}
        QPushButton#btn-primary:hover {{
            background: rgb({greenDark.red()},{greenDark.green()},{greenDark.blue()});
        }}
        QPushButton#btn-primary:pressed {{
            background: rgb({greenDark.darker(115).red()},{greenDark.darker(115).green()},{greenDark.darker(115).blue()});
        }}
        QPushButton#btn-primary:focus {{
            outline: none;
            box-shadow: 0 0 0 3px rgba(34,139,34,0.25);
        }}

        /* Ghost (secundario claro) */
        QPushButton#btn-ghost {{
            background: transparent;
            border: 1px dashed rgba(0,0,0,22%);
            color: rgb({text.red()},{text.green()},{text.blue()});
        }}
        QPushButton#btn-ghost:hover {{
            background: rgba({green.red()},{green.green()},{green.blue()}, 0.08);
            border-color: rgba({green.red()},{green.green()},{green.blue()}, 0.35);
        }}
        QPushButton#btn-ghost:pressed {{
            background: rgba({green.red()},{green.green()},{green.blue()}, 0.14);
        }}

        /* Danger (rojo para acciones fuertes, por si lo usas) */
        QPushButton#btn-danger {{
            background: #d9534f;
            color: white;
            border: 1px solid rgba(0,0,0,10%);
            font-weight: 600;
        }}
        QPushButton#btn-danger:hover {{ background: #c94541; }}

        /* Entradas de texto */
        QLineEdit, QPlainTextEdit, QTextEdit {{
            background: rgb({panel.red()},{panel.green()},{panel.blue()});
            border: 1px solid rgba(0,0,0,16%);
            border-radius: 8px;
            padding: 6px 8px;
        }}
        QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
            border: 1px solid rgba({green.red()},{green.green()},{green.blue()}, 0.9);
            box-shadow: 0 0 0 3px rgba({green.red()},{green.green()},{green.blue()}, 0.15);
        }}

        /* TabWidget */
        QTabWidget::pane {{
            border: 1px solid rgba(0,0,0,12%);
            border-radius: 10px;
            padding: 6px;
            background: rgb({panel.red()},{panel.green()},{panel.blue()});
        }}
        QTabBar::tab {{
            background: transparent;
            border: none;
            padding: 8px 14px;
            margin: 4px;
            border-radius: 8px;
            color: {subtext.name()};
            font-weight: 600;
        }}
        QTabBar::tab:selected {{
            color: white;
            background: rgb({green.red()},{green.green()},{green.blue()});
        }}
        QTabBar::tab:hover:!selected {{
            color: rgb({green.darker(120).red()},{green.darker(120).green()},{green.darker(120).blue()});
            background: rgba({green.red()},{green.green()},{green.blue()}, 0.08);
        }}

        /* Splitter */
        QSplitter::handle {{
            background: rgba(0,0,0,6%);
            width: 6px;
            margin: 2px;
            border-radius: 3px;
        }}

        /* Tablas */
        QTableWidget {{
            border: 1px solid rgba(0,0,0,12%);
            border-radius: 8px;
            gridline-color: rgba(0,0,0,10%);
            selection-background-color: rgb({green.red()},{green.green()},{green.blue()});
            selection-color: white;
        }}
        QHeaderView::section {{
            background: rgba({green.red()},{green.green()},{green.blue()}, 0.10);
            color: rgb({text.red()},{text.green()},{text.blue()});
            border: none;
            border-right: 1px solid rgba(0,0,0,10%);
            padding: 6px;
            font-weight: 600;
        }}

        /* SpinBox y ComboBox */
        QSpinBox, QDoubleSpinBox, QComboBox {{
            background: rgb({panel.red()},{panel.green()},{panel.blue()});
            border: 1px solid rgba(0,0,0,16%);
            border-radius: 8px;
            padding: 4px 8px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}

        /* Ocultar flechas de todos los SpinBox */
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 0px; height: 0px; border: none; margin: 0; padding: 0;
        }}

        /* Áreas de salida (OutputArea) ligeramente resaltadas */
        QTextEdit[readOnly="true"] {{
            background: rgb({panel.red()},{panel.green()},{panel.blue()});
            border: 1px solid rgba(0,0,0,12%);
            border-radius: 10px;
        }}

        /* Barras de desplazamiento finas */
        QScrollBar:vertical {{
            width: 10px; background: transparent; margin: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(0,0,0,20%); border-radius: 5px; min-height: 24px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(0,0,0,32%);
        }}
        QScrollBar:horizontal {{
            height: 10px; background: transparent; margin: 4px;
        }}
        QScrollBar::handle:horizontal {{
            background: rgba(0,0,0,20%); border-radius: 5px; min-width: 24px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: rgba(0,0,0,32%);
        }}
    """)


# =========================
# 4. VENTANA PRINCIPAL (El ensamblador)
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculadora de Álgebra Lineal y Métodos Numéricos")
        self.resize(1200, 820)

        tabs = QtWidgets.QTabWidget()
        
        # EL ENSAMBLAJE FINAL DE LOS MÓDULOS IMPORTADOS (Las 8 pestañas)
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

        # Se asume que esta función se importó de widgets.py
        hide_all_spin_buttons(self)

# =========================
# 5. PUNTO DE ENTRADA
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_green_theme(app)  # Aplicar el tema al iniciar la aplicación

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()