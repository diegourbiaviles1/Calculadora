# app_gui.py — GUI PyQt6 completa (incluye Programa 6: Inversa por Gauss–Jordan)
from __future__ import annotations
from typing import List
import sys
import re
from fractions import Fraction

from PyQt6 import QtWidgets, QtCore, QtGui

from utilidad import evaluar_expresion, fmt_number, DEFAULT_DEC
from sistema_lineal import SistemaLineal, formatear_solucion_parametrica
from homogeneo import (
    resolver_sistema_homogeneo_y_no_homogeneo,
    resolver_dependencia_lineal_con_homogeneo,
    analizar_dependencia,
)
from algebra_vector import (
    resolver_AX_igual_B,
    multiplicacion_matriz_vector_explicada,
    sistema_a_forma_matricial,
    ecuacion_vectorial,
    combinacion_lineal_explicada,
    verificar_propiedades,
    verificar_distributiva_matriz,
)

# ====== Funciones del Programa 5 ======
from matrices import (
    suma_matrices_explicada,
    resta_matrices_explicada,
    producto_escalar_explicado,
    producto_matrices_explicado,
    traspuesta_explicada,
    propiedad_r_suma_traspuesta_explicada, 
)

# ====== Programa 6 (Inversa) ======
from inversa import (
    inversa_por_gauss_jordan,
    verificar_propiedades_invertibilidad,
    programa_inversa_con_propiedades,
)

# =========================
#   Helpers de parsing
# =========================
def parse_nums(line: str) -> List[float]:
    line = (line or "").replace("|", " ")
    parts = [p for p in line.replace(",", " ").split() if p]
    return [float(evaluar_expresion(p, exacto=False)) for p in parts]

def parse_vector(text: str, n: int) -> List[float]:
    vals = parse_nums(text)
    if len(vals) != n:
        raise ValueError(f"Se esperaban {n} valores, recibidos {len(vals)}.")
    return vals

def parse_matrix(text: str, m: int, n: int) -> List[List[float]]:
    rows = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(rows) != m:
        raise ValueError(f"Se esperaban {m} filas, recibidas {len(rows)}.")
    A = []
    for i, ln in enumerate(rows, 1):
        nums = parse_nums(ln)
        if len(nums) != n:
            raise ValueError(f"Fila {i}: se esperaban {n} valores, recibidos {len(nums)}.")
        A.append(nums)
    return A

def parse_augmented(text: str, m: int, n: int) -> List[List[float]]:
    rows = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(rows) != m:
        raise ValueError(f"Se esperaban {m} filas, recibidas {len(rows)}.")
    Ab = []
    for i, ln in enumerate(rows, 1):
        nums = parse_nums(ln)
        if len(nums) != n + 1:
            raise ValueError(f"Fila {i}: se esperaban {n+1} valores (coeficientes y b).")
        Ab.append(nums)
    return Ab

def parse_list_vectors(text: str, k: int, n: int) -> List[List[float]]:
    rows = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(rows) != k:
        raise ValueError(f"Se esperaban {k} líneas, recibidas {len(rows)}.")
    out = []
    for i, ln in enumerate(rows, 1):
        nums = parse_nums(ln)
        if len(nums) != n:
            raise ValueError(f"Vector {i}: se esperaban {n} valores, recibidos {len(nums)}.")
        out.append(nums)
    return out

def pretty_augmented_from_table(table: "MatrixAugTable", dec: int = 4) -> str:
    m = table.table.rowCount()
    n = table.table.columnCount() - 1
    grid = []
    for i in range(m):
        row = []
        for j in range(n + 1):
            it = table.table.item(i, j)
            row.append((it.text() if it else "0").strip())
        grid.append(row)
    widths = [0] * (n + 1)
    for j in range(n + 1):
        widths[j] = max(len(grid[i][j]) if grid[i][j] else 1 for i in range(m))
    rows_txt = []
    for i in range(m):
        left = "  ".join(grid[i][j].rjust(widths[j]) for j in range(n))
        right = grid[i][n].rjust(widths[n])
        rows_txt.append(f"{left}  |  {right}" if n > 0 else right)
    L = ["⎡", "⎢", "⎣"]
    R = ["⎤", "⎥", "⎦"]
    out = []
    for i, row in enumerate(rows_txt):
        lbr = L[0] if i == 0 else (L[2] if i == m - 1 else L[1])
        rbr = R[0] if i == 0 else (R[2] if i == m - 1 else R[1])
        out.append(f"{lbr} {row} {rbr}")
    return "\n".join(out)

# ===== Helpers extra GUI =====
def fill_augmented_template(m: int, n: int) -> str:
    fila = " ".join("0" for _ in range(n)) + " | 0"
    return "\n".join(fila for _ in range(m))

def demo_augmented_3x3() -> str:
    return "\n".join([
        "2  1 -1 |  8",
        "-3 -1  2 | -11",
        "-2  1  2 | -3",
    ])

def save_text_to_file(parent: QtWidgets.QWidget, text: str, suggested_name="salida.txt"):
    if not text.strip():
        QtWidgets.QMessageBox.information(parent, "Exportar", "No hay nada para exportar.")
        return
    fn, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Guardar salida", suggested_name, "Texto (*.txt)")
    if fn:
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)

# ===== Helpers de tabla =====
def _safe_float(x: str) -> float:
    x = (x or "").strip()
    if not x:
        return 0.0
    return float(evaluar_expresion(x, exacto=False))

def table_to_augmented(table: "MatrixAugTable") -> list[list[float]]:
    m = table.table.rowCount()
    n = table.table.columnCount() - 1
    Ab = []
    for i in range(m):
        fila = []
        for j in range(n + 1):
            item = table.table.item(i, j)
            txt = "" if item is None else item.text()
            fila.append(_safe_float(txt))
        Ab.append(fila)
    return Ab

# =========================
#   Widgets básicos
# =========================
def mono_font():
    f = QtGui.QFont("Consolas")
    f.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
    f.setPointSize(10)
    return f

def _disable_spin_wheel(sb: QtWidgets.QAbstractSpinBox):
    # evita que cambie el valor con la rueda del mouse
    def _no_wheel(event): 
        event.ignore()
    sb.wheelEvent = _no_wheel  # type: ignore

def hide_all_spin_buttons(root: QtWidgets.QWidget):
    """
    Oculta los botones ↑↓ de TODOS los spin boxes del árbol y desactiva la rueda.
    """
    for sb in root.findChildren(QtWidgets.QAbstractSpinBox):
        sb.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        _disable_spin_wheel(sb)

def apply_green_theme(app: QtWidgets.QApplication):
    """
    Tema verde moderno: botones con hover/focus, tabs marcados en verde,
    inputs y tablas con acentos verdes y esquinas redondeadas.
    """
    # ---- Paleta base clara con Highlights en verde ----
    palette = QtGui.QPalette()
    bg        = QtGui.QColor(246, 248, 246)       # casi blanco con tinte verde
    panel     = QtGui.QColor(255, 255, 255)
    text      = QtGui.QColor(28, 28, 28)
    subtext   = QtGui.QColor(95, 95, 95)
    green     = QtGui.QColor(34, 139, 34)         # forest green
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

        /* 🔒 Ocultar flechas de todos los SpinBox */
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


def btn(text: str, kind: str = "primary"):
    b = QtWidgets.QPushButton(text)
    b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    # Usa objectName para que el QSS los detecte
    if kind not in {"primary", "ghost", "danger"}:
        kind = "primary"
    b.setObjectName(f"btn-{kind}")
    b.setMinimumHeight(34)
    b.setProperty("class", kind)
    b.setAutoDefault(False)
    b.setDefault(False)
    return b

class LabeledEdit(QtWidgets.QWidget):
    def __init__(self, label, placeholder="", parent=None, default_value=""):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.lbl = QtWidgets.QLabel(label)
        self.edit = QtWidgets.QLineEdit(); self.edit.setPlaceholderText(placeholder)
        if default_value:
            self.edit.setText(default_value)
        lay.addWidget(self.lbl); lay.addWidget(self.edit)

    def text(self): return self.edit.text().strip()
    def setText(self, t): self.edit.setText(t)

class MatrixInput(QtWidgets.QWidget):
    """Entrada multilinea para matrices/vectores con tipografía monoespaciada."""
    def __init__(self, title="Matriz", height=120, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.lbl = QtWidgets.QLabel(title)
        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setFont(mono_font()); self.txt.setFixedHeight(height)
        lay.addWidget(self.lbl); lay.addWidget(self.txt)

    def text(self): return self.txt.toPlainText().strip()
    def setText(self, t): self.txt.setPlainText(t or "")

class OutputArea(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(mono_font())
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def clear_and_write(self, s: str):
        self.clear(); self.setPlainText(s or "")

# =========================
#   Conversores de formato (decimales ↔ fracciones)
# =========================
_frac_re = re.compile(r'(?<![\w])(-?\d+)\s*/\s*(-?\d+)(?![\w])')
_dec_re  = re.compile(r'(?<![\w/.-])(-?(?:\d+\.\d+|\d+))(?![\w/])')

def _to_fraction_str(x: float) -> str:
    fr = Fraction(x).limit_denominator(1000000)
    if fr.denominator == 1:
        return f"{fr.numerator}"
    return f"{fr.numerator}/{fr.denominator}"

def text_to_fractions(text: str) -> str:
    def repl_dec(m):
        s = m.group(1)
        try:
            val = float(s)
        except:
            return s
        return _to_fraction_str(val)
    return _dec_re.sub(repl_dec, text)

def text_to_decimals(text: str, dec: int = DEFAULT_DEC) -> str:
    def repl_frac(m):
        num = int(m.group(1)); den = int(m.group(2))
        if den == 0:
            return f"{num}/0"
        val = num / den
        return fmt_number(val, dec, False)
    return _frac_re.sub(repl_frac, text)

# =========================
#   Editor tabular Ab
# =========================
class MatrixAugTable(QtWidgets.QWidget):
    """
    Editor para matrices aumentadas (m x (n+1)).
    Añadimos set_size() y to_text() para integrarlo con otras pestañas.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)

        # Barra superior: m, n
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("m:"))
        self.spin_m = QtWidgets.QSpinBox(); self.spin_m.setRange(1, 999); self.spin_m.setValue(3)
        self.spin_m.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        _disable_spin_wheel(self.spin_m)
        top.addWidget(self.spin_m)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("n:"))
        self.spin_n = QtWidgets.QSpinBox(); self.spin_n.setRange(1, 999); self.spin_n.setValue(3)
        self.spin_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        _disable_spin_wheel(self.spin_n)
        top.addWidget(self.spin_n)
        top.addStretch(1)
        lay.addLayout(top)

        # Tabla
        self.table = QtWidgets.QTableWidget(3, 4)  # 3 x (3+1)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.table.verticalHeader().setVisible(False)

        self._refresh_headers()
        lay.addWidget(self.table)

        # Conexiones
        self.spin_m.valueChanged.connect(self._resize_table)
        self.spin_n.valueChanged.connect(self._resize_table)

    def _refresh_headers(self):
        n = self.table.columnCount() - 1
        labels = [f"X{j+1}" for j in range(n)]
        labels.append("B")
        self.table.setHorizontalHeaderLabels(labels)

    def _resize_table(self):
        m = int(self.spin_m.value())
        n = int(self.spin_n.value())
        if self.table.rowCount() != m:
            self.table.setRowCount(m)
        if self.table.columnCount() != (n + 1):
            self.table.setColumnCount(n + 1)
            self._refresh_headers()

    # API extra
    def set_size(self, m: int, n: int):
        self.spin_m.setValue(m)
        self.spin_n.setValue(n)
        self._resize_table()

    def to_text(self) -> str:
        m = self.table.rowCount()
        n = self.table.columnCount() - 1
        lines = []
        for i in range(m):
            row = []
            for j in range(n + 1):
                it = self.table.item(i, j)
                row.append((it.text() if it else "0").strip() or "0")
            left = " ".join(row[:-1]); right = row[-1]
            lines.append(f"{left} | {right}" if n > 0 else right)
        return "\n".join(lines)

    def fill_zeros(self):
        m, n = self.table.rowCount(), self.table.columnCount()
        for i in range(m):
            for j in range(n):
                item = self.table.item(i, j)
                if item is None:
                    item = QtWidgets.QTableWidgetItem("0")
                    self.table.setItem(i, j, item)
                else:
                    item.setText("0")

    def clear_all(self):
        m, n = self.table.rowCount(), self.table.columnCount()
        for i in range(m):
            for j in range(n):
                item = self.table.item(i, j)
                if item is None:
                    item = QtWidgets.QTableWidgetItem("")
                    self.table.setItem(i, j, item)
                else:
                    item.setText("")

# =========================
#   Tabla genérica para matrices (NO aumentada)
# =========================
class MatrixTable(QtWidgets.QWidget):
    """Tabla genérica de tamaño m x n para matrices."""
    def __init__(self, m=2, n=2, title="Matriz", parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.lbl = QtWidgets.QLabel(title); lay.addWidget(self.lbl)
        self.table = QtWidgets.QTableWidget(m, n)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self._refresh_headers()
        lay.addWidget(self.table)

    def set_title(self, t): self.lbl.setText(t)

    def _refresh_headers(self):
        n = self.table.columnCount()
        self.table.setHorizontalHeaderLabels([f"C{j+1}" for j in range(n)])

    def set_size(self, m, n):
        if self.table.rowCount() != m:
            self.table.setRowCount(m)
        if self.table.columnCount() != n:
            self.table.setColumnCount(n)
            self._refresh_headers()

    def fill_zeros(self):
        m, n = self.table.rowCount(), self.table.columnCount()
        for i in range(m):
            for j in range(n):
                it = self.table.item(i, j)
                if it is None:
                    it = QtWidgets.QTableWidgetItem("0")
                    self.table.setItem(i, j, it)
                else:
                    it.setText("0")

    def clear_all(self):
        m, n = self.table.rowCount(), self.table.columnCount()
        for i in range(m):
            for j in range(n):
                it = self.table.item(i, j)
                if it is None:
                    self.table.setItem(i, j, QtWidgets.QTableWidgetItem(""))
                else:
                    it.setText("")

    def to_matrix(self) -> list[list[float]]:
        m, n = self.table.rowCount(), self.table.columnCount()
        A = []
        for i in range(m):
            row = []
            for j in range(n):
                t = self.table.item(i, j).text() if self.table.item(i, j) else "0"
                row.append(_safe_float(t))
            A.append(row)
        return A
    
# --- NUEVO WIDGET para entrada de vectores ---
class VectorInputTable(QtWidgets.QWidget):
    """Un widget para introducir un vector en una tabla de una sola fila."""
    def __init__(self, title: str, n_initial: int = 3, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(title)
        self.table = QtWidgets.QTableWidget(1, n_initial)
        self.table.setFixedHeight(55)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.label)
        layout.addWidget(self.table)
        self._refresh_headers()

    def _refresh_headers(self):
        self.table.setHorizontalHeaderLabels([f"v{j+1}" for j in range(self.table.columnCount())])
        
    def set_size(self, n: int):
        if self.table.columnCount() != n:
            self.table.setColumnCount(n)
            self._refresh_headers()

    def to_vector(self) -> list[float]:
        return [_safe_float(self.table.item(0, j).text() if self.table.item(0, j) else "0") for j in range(self.table.columnCount())]

# ===== Formateo de matrices para salida en texto =====
def format_matrix_text(M, dec=DEFAULT_DEC) -> str:
    if not M:
        return "[ ]"
    n = len(M[0])
    col_w = [0]*n
    grid = [[fmt_number(x, dec, False) for x in fila] for fila in M]
    for j in range(n):
        col_w[j] = max(len(grid[i][j]) for i in range(len(M)))
    lines = []
    for fila in grid:
        parts = [s.rjust(col_w[j]) for j, s in enumerate(fila)]
        lines.append("[ " + "  ".join(parts) + " ]")
    return "\n".join(lines)

# ===== Explicación elemento a elemento =====
def _num(x):  # usa el mismo formato que el resto de la app
    return fmt_number(x, DEFAULT_DEC, False)

def explain_sum(A, B):
    m, n = len(A), len(A[0])
    lines = []
    for i in range(m):
        for j in range(n):
            a, b = A[i][j], B[i][j]
            lines.append(f"C[{i+1},{j+1}] = A[{i+1},{j+1}] + B[{i+1},{j+1}] = "
                         f"{_num(a)} + {_num(b)} = {_num(a+b)}")
    return lines

def explain_res(A, B):
    m, n = len(A), len(A[0])
    lines = []
    for i in range(m):
        for j in range(n):
            a, b = A[i][j], B[i][j]
            lines.append(f"C[{i+1},{j+1}] = A[{i+1},{j+1}] - B[{i+1},{j+1}] = "
                         f"{_num(a)} - {_num(b)} = {_num(a-b)}")
    return lines

def explain_kA(k, A):
    m, n = len(A), len(A[0])
    lines = []
    for i in range(m):
        for j in range(n):
            a = A[i][j]
            lines.append(f"C[{i+1},{j+1}] = k·A[{i+1},{j+1}] = {_num(k)}·{_num(a)} = {_num(k*a)}")
    return lines

def explain_AB(A, B):
    m, p = len(A), len(B[0])
    n = len(A[0])  # = filas(B)
    lines = []
    for i in range(m):
        for j in range(p):
            terms = [f"A[{i+1},{k+1}]·B[{k+1},{j+1}]={_num(A[i][k])}·{_num(B[k][j])}"
                     for k in range(n)]
            s = sum(A[i][k]*B[k][j] for k in range(n))
            lines.append(
                f"C[{i+1},{j+1}] = " +
                " + ".join(terms) +
                f" = {_num(s)}"
            )
    return lines

# =========================
#   Pestañas existentes
# =========================
class TabGaussJordan(QtWidgets.QWidget):
    """Resolver por Gauss–Jordan a partir de matriz aumentada (tabla a la izquierda, vista inicial a la derecha)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)

        self.group = QtWidgets.QGroupBox("Tamaño de la matriz")
        gl = QtWidgets.QVBoxLayout(self.group)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left); left_lay.setContentsMargins(0,0,0,0)
        self.aug_table = MatrixAugTable()
        left_lay.addWidget(self.aug_table)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right); right_lay.setContentsMargins(6,0,0,0)
        self.lbl_preview = QtWidgets.QLabel("Matriz aumentada inicial")
        self.lbl_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.preview = OutputArea()
        self.preview.setReadOnly(True)
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        right_lay.addWidget(self.lbl_preview)
        right_lay.addWidget(self.preview)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        gl.addWidget(splitter)
        mlay.addWidget(self.group)

        btn_bar = QtWidgets.QHBoxLayout()
        self.btn_zeros = QtWidgets.QPushButton("Rellenar ceros")
        self.btn_clear = QtWidgets.QPushButton("Limpiar")
        self.btn_solve = QtWidgets.QPushButton("Resolver por Gauss Jordan")
        btn_bar.addStretch(1)
        btn_bar.addWidget(self.btn_zeros)
        btn_bar.addWidget(self.btn_clear)
        btn_bar.addWidget(self.btn_solve)
        btn_bar.addStretch(1)
        gl.addLayout(btn_bar)

        self.lbl_procedimiento = QtWidgets.QLabel("Procedimiento (reducción por filas)")
        self.lbl_procedimiento.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        mlay.addWidget(self.lbl_procedimiento)

        self.out = OutputArea()
        mlay.addWidget(self.out)

        # === Botón global de formato (pestaña) ===
        self._show_frac = False
        self._last_dec = ""
        self._last_frac = ""
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_fmt.clicked.connect(self._toggle_fmt)

        self.btn_zeros.clicked.connect(self.aug_table.fill_zeros)
        self.btn_clear.clicked.connect(self.aug_table.clear_all)
        self.btn_solve.clicked.connect(self.on_run_table)

        self.aug_table.table.itemChanged.connect(self.update_preview)
        self.aug_table.spin_m.valueChanged.connect(lambda _: self.update_preview())
        self.aug_table.spin_n.valueChanged.connect(lambda _: self.update_preview())

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self.on_run_table)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self.on_run_table)

        self.update_preview()

    def update_preview(self):
        self.preview.clear_and_write(pretty_augmented_from_table(self.aug_table))

    def _set_text(self, base_text: str):
        self._last_dec = text_to_decimals(base_text)
        self._last_frac = text_to_fractions(base_text)
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _render_out(self, out: dict):
        lines = ["=== Pasos ==="]
        for p in out.get("pasos", []):
            lines.append(p + "\n")
        if out["tipo"] == "unica":
            lines.append("Solución única x = " + str(out["x"]))
        elif out["tipo"] == "infinitas":
            lines.append("Infinitas soluciones. Variables libres en columnas: " + str(out["libres"]))
        else:
            lines.append("Sistema inconsistente (sin solución).")
        lines.append("")
        param_txt = formatear_solucion_parametrica(out, nombres_vars=None, dec=4, fracciones=True)
        self._set_text("\n".join(lines + [param_txt]))

    def on_run_table(self):
        try:
            Ab = table_to_augmented(self.aug_table)
            sl = SistemaLineal(Ab, decimales=4)
            out = sl.gauss_jordan()
            self._render_out(out)
        except Exception as e:
            self._set_text("Error: " + str(e))

class TabProg4(QtWidgets.QWidget):
    """Dependencia: Ax=b (pasos+paramétrica) y dependencia A·c=0."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)

        # --- Menú de selección de operación ---
        top_bar = QtWidgets.QHBoxLayout()
        self.op_selector = QtWidgets.QComboBox()
        self.op_selector.addItems([
            "Resolver Ax=b + Dependencia de columnas de A",
            "Análisis de Dependencia Lineal (de un conjunto de vectores)"
        ])
        # (Eliminado botón Exportar resultados…)
        top_bar.addWidget(QtWidgets.QLabel("Operación:"))
        top_bar.addWidget(self.op_selector, 1)
        mlay.addLayout(top_bar)
        
        # --- Contenedor de páginas ---
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._create_axb_page())
        self.stack.addWidget(self._create_dependencia_page())
        mlay.addWidget(self.stack)
        
        self.out = OutputArea()
        mlay.addWidget(self.out)

        # === Botón global de formato (pestaña) ===
        self._show_frac = False
        self._last_dec = ""
        self._last_frac = ""
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_fmt.clicked.connect(self._toggle_fmt)
        
        self.op_selector.currentIndexChanged.connect(self.stack.setCurrentIndex)

    def _set_text(self, base_text: str):
        self._last_dec = text_to_decimals(base_text)
        self._last_frac = text_to_fractions(base_text)
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _create_axb_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        dims = QtWidgets.QHBoxLayout()
        self.m_in_axb = LabeledEdit("m (filas):", "3", default_value="3")
        self.n_in_axb = LabeledEdit("n (columnas):", "3", default_value="3")
        dims.addWidget(self.m_in_axb)
        dims.addWidget(self.n_in_axb)
        layout.addLayout(dims)
        
        self.A_table_axb = MatrixTable(title="Matriz A (m x n)")
        self.b_input_axb = VectorInputTable("Vector b (m valores)", n_initial=3)
        layout.addWidget(self.A_table_axb)
        layout.addWidget(self.b_input_axb)
        
        run_btn = btn("Resolver Sistema y Analizar Dependencia de A")
        run_btn.clicked.connect(self.on_run_axb)
        layout.addWidget(run_btn)
        
        self.m_in_axb.edit.textChanged.connect(self._sync_axb)
        self.n_in_axb.edit.textChanged.connect(self._sync_axb)
        self._sync_axb()
        return page

    def _sync_axb(self):
        try:
            m, n = int(self.m_in_axb.text()), int(self.n_in_axb.text())
            self.A_table_axb.set_size(m, n)
            self.b_input_axb.set_size(m)
        except ValueError: pass

    def _create_dependencia_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        dims = QtWidgets.QHBoxLayout()
        self.k_in_dep = LabeledEdit("k (n° vectores):", "3", default_value="3")
        self.n_in_dep = LabeledEdit("n (dimensión):", "3", default_value="3")
        dims.addWidget(self.k_in_dep)
        dims.addWidget(self.n_in_dep)
        layout.addLayout(dims)
        
        self.V_table_dep = MatrixTable(title="Lista de Vectores v₁..vₖ (k filas, n columnas)")
        layout.addWidget(self.V_table_dep)
        
        run_btn = btn("Analizar Dependencia Lineal de los Vectores")
        run_btn.clicked.connect(self.on_run_dependencia)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        
        self.k_in_dep.edit.textChanged.connect(self._sync_dep)
        self.n_in_dep.edit.textChanged.connect(self._sync_dep)
        self._sync_dep()
        return page

    def _sync_dep(self):
        try:
            k, n = int(self.k_in_dep.text()), int(self.n_in_dep.text())
            self.V_table_dep.set_size(k, n)
        except ValueError: pass

    def on_run_axb(self):
        try:
            A = self.A_table_axb.to_matrix()
            b_vals = self.b_input_axb.to_vector()
            info = resolver_sistema_homogeneo_y_no_homogeneo(A, b_vals)
            
            lines = ["=== PASOS (Gauss-Jordan aplicado a Ax = b) ==="] + info.get("pasos", [])
            lines += ["\n=== SOLUCIÓN GENERAL (forma paramétrica) ===", info["salida_parametrica"]]
            lines += ["\n=== CONCLUSIÓN DEL SISTEMA ===", info["conclusion"]]
            
            dep = resolver_dependencia_lineal_con_homogeneo(A)
            lines += ["\n\n=== ANÁLISIS DE DEPENDENCIA LINEAL (Columnas de A) ===", dep.get("dependencia", "(sin análisis)")]
            lines += ["\n=== PASOS (Gauss-Jordan aplicado a A·c = 0) ==="] + dep.get("pasos", [])
            lines += ["\n=== COMBINACIÓN LINEAL (forma paramétrica de los coeficientes c) ===", dep.get("salida_parametrica", "")]
            self._set_text("\n".join(lines))
        except Exception as e:
            self._set_text(f"Error: {e}")

    def on_run_dependencia(self):
        try:
            V = self.V_table_dep.to_matrix()
            info = analizar_dependencia(V)
            lines = ["=== Conclusión ===", info["mensaje"]]
            lines += ["\n=== Pasos (Gauss-Jordan sobre A·c = 0, donde las columnas de A son los vectores vᵢ) ===\n"] + [p + "\n" for p in info["pasos"]]
            lines += ["\n=== Coeficientes 'c' para la combinación lineal nula ===", info["salida_parametrica"]]
            self._set_text("\n".join(lines))
        except Exception as e:
            self._set_text(f"Error: {e}")

class TabAXeqB(QtWidgets.QWidget):
    """Resolver AX=B (B vector o matriz)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)

        dims = QtWidgets.QHBoxLayout()
        self.m_in = LabeledEdit("m (filas):", "3", default_value="3")
        self.n_in = LabeledEdit("n (columnas):", "3", default_value="3")
        dims.addWidget(self.m_in)
        dims.addWidget(self.n_in)
        mlay.addLayout(dims)

        self.A_table = MatrixTable(title="Matriz A (m x n)")
        self.b_input = VectorInputTable("Vector b (m valores)", n_initial=3)
        mlay.addWidget(self.A_table)
        mlay.addWidget(self.b_input)

        self.run = btn("Resolver AX = b")
        mlay.addWidget(self.run)

        self.out = OutputArea()
        mlay.addWidget(self.out)

        # === Botón global de formato (pestaña) ===
        self._show_frac = False
        self._last_dec = ""
        self._last_frac = ""
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_fmt.clicked.connect(self._toggle_fmt)

        self.run.clicked.connect(self.on_run)

        self.m_in.edit.textChanged.connect(self._sync_dims)
        self.n_in.edit.textChanged.connect(self._sync_dims)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self.on_run)
        
        self._sync_dims()

    def _sync_dims(self):
        try:
            m = int(self.m_in.text())
            n = int(self.n_in.text())
            self.A_table.set_size(m, n)
            self.b_input.set_size(m)
        except ValueError:
            pass

    def _set_text(self, base_text: str):
        self._last_dec = text_to_decimals(base_text)
        self._last_frac = text_to_fractions(base_text)
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def on_run(self):
        try:
            A = self.A_table.to_matrix()
            b_vals = self.b_input.to_vector()
            
            out = resolver_AX_igual_B(A, b_vals)
            lines = []
            lines.append("--- Procedimiento de Gauss-Jordan ---")
            for p in out.get("reportes", []): lines.append(p)
            
            lines.append("\n--- Resultado ---")
            if out.get("estado") == "ok":
                if "x" in out:
                    lines.append("Solución única encontrada:")
                    lines.append("x = " + str(out["x"]))
                if "X" in out:
                    lines.append("Matriz solución X encontrada:")
                    lines.append(format_matrix_text(out["X"]))
            else:
                lines.append("Estado del sistema: " + str(out.get("estado")))
                lines.append("No se encontró una solución única.")
            self._set_text("\n".join(lines))
        except Exception as e:
            self._set_text("Error: " + str(e))

class TabVectores(QtWidgets.QWidget):
    """Herramientas: ecuación vectorial y combinación lineal."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)
        
        # --- Menú de selección de operación ---
        top_bar = QtWidgets.QHBoxLayout()
        self.op_selector = QtWidgets.QComboBox()
        self.op_selector.addItems([
            "1) Verificar propiedades en Rⁿ",
            "2) Verificar distributiva A(u+v) = Au+Av",
            "3) Combinación lineal de vectores",
            "4) Ecuación vectorial (¿b en span{v₁..vₖ}?)"
        ])
        top_bar.addWidget(QtWidgets.QLabel("Seleccione una operación:"))
        top_bar.addWidget(self.op_selector, 1)
        mlay.addLayout(top_bar)
        
        # --- Contenedor de páginas ---
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._create_prop_rn_page())
        self.stack.addWidget(self._create_distributiva_page())
        self.stack.addWidget(self._create_comb_lineal_page())
        self.stack.addWidget(self._create_ecu_vec_page())
        mlay.addWidget(self.stack)
        
        self.out = OutputArea()
        mlay.addWidget(self.out)

        # === Botón global de formato (pestaña) ===
        self._show_frac = False
        self._last_dec = ""
        self._last_frac = ""
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_fmt.clicked.connect(self._toggle_fmt)
        
        self.op_selector.currentIndexChanged.connect(self.stack.setCurrentIndex)
    
    # helpers de formato
    def _set_text(self, base_text: str):
        self._last_dec = text_to_decimals(base_text)
        self._last_frac = text_to_fractions(base_text)
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)
    
    # --- Página 1: Propiedades en R^n ---
    def _create_prop_rn_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.n_in_prn = LabeledEdit("Dimensión n:", "3", default_value="3")
        self.v_in_prn = VectorInputTable("Vector v", 3)
        self.u_in_prn = VectorInputTable("Vector u", 3)
        self.w_in_prn = VectorInputTable("Vector w", 3)
        scalars_layout = QtWidgets.QHBoxLayout()
        self.a_in_prn = LabeledEdit("Escalar a:", "1.5", default_value="1.5")
        self.b_in_prn = LabeledEdit("Escalar b:", "-2", default_value="-2")
        scalars_layout.addWidget(self.a_in_prn)
        scalars_layout.addWidget(self.b_in_prn)
        run_btn = btn("Verificar Propiedades")
        run_btn.clicked.connect(self.on_run_prop_rn)
        layout.addWidget(self.n_in_prn)
        layout.addWidget(self.v_in_prn)
        layout.addWidget(self.u_in_prn)
        layout.addWidget(self.w_in_prn)
        layout.addLayout(scalars_layout)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        self.n_in_prn.edit.textChanged.connect(self._sync_prop_rn)
        self._sync_prop_rn()
        return page

    def _sync_prop_rn(self):
        try:
            n = int(self.n_in_prn.text())
            self.v_in_prn.set_size(n)
            self.u_in_prn.set_size(n)
            self.w_in_prn.set_size(n)
        except ValueError: pass

    def on_run_prop_rn(self):
        try:
            v = self.v_in_prn.to_vector()
            u = self.u_in_prn.to_vector()
            w = self.w_in_prn.to_vector()
            a = float(evaluar_expresion(self.a_in_prn.text()))
            b = float(evaluar_expresion(self.b_in_prn.text()))
            res = verificar_propiedades(v, u, w, a, b)
            lines = ["--- Verificación de Propiedades ---"]
            for k, val in res.items():
                lines.append(f"{k.replace('_', ' '):<20}: {'✔️ Cumplida' if val else '❌ No Cumplida'}")
            self._set_text("\n".join(lines))
        except Exception as e:
            self._set_text(f"Error: {e}")

    # --- Página 2: Distributiva ---
    def _create_distributiva_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        dims = QtWidgets.QHBoxLayout()
        self.m_in_dist = LabeledEdit("Filas de A (m):", "2", default_value="2")
        self.n_in_dist = LabeledEdit("Columnas de A (n):", "3", default_value="3")
        dims.addWidget(self.m_in_dist)
        dims.addWidget(self.n_in_dist)
        self.A_in_dist = MatrixTable(2, 3, "Matriz A")
        self.u_in_dist = VectorInputTable("Vector u (tamaño n)", 3)
        self.v_in_dist = VectorInputTable("Vector v (tamaño n)", 3)
        run_btn = btn("Verificar Propiedad Distributiva")
        run_btn.clicked.connect(self.on_run_distributiva)
        layout.addLayout(dims)
        layout.addWidget(self.A_in_dist)
        layout.addWidget(self.u_in_dist)
        layout.addWidget(self.v_in_dist)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        self.m_in_dist.edit.textChanged.connect(self._sync_dist)
        self.n_in_dist.edit.textChanged.connect(self._sync_dist)
        self._sync_dist()
        return page

    def _sync_dist(self):
        try:
            m, n = int(self.m_in_dist.text()), int(self.n_in_dist.text())
            self.A_in_dist.set_size(m, n)
            self.u_in_dist.set_size(n)
            self.v_in_dist.set_size(n)
        except ValueError: pass

    def on_run_distributiva(self):
        try:
            A = self.A_in_dist.to_matrix()
            u = self.u_in_dist.to_vector()
            v = self.v_in_dist.to_vector()
            out = verificar_distributiva_matriz(A, u, v)
            self._set_text("\n".join(out["pasos"]))
        except Exception as e:
            self._set_text("Error: " + str(e))

    # --- Página 3: Combinación Lineal ---
    def _create_comb_lineal_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        dims = QtWidgets.QHBoxLayout()
        self.k_in_cl = LabeledEdit("k (n° vectores):", "2", default_value="2")
        self.n_in_cl = LabeledEdit("n (dimensión):", "3", default_value="3")
        dims.addWidget(self.k_in_cl)
        dims.addWidget(self.n_in_cl)
        self.V_in_cl = MatrixTable(2, 3, "Lista de vectores v₁..vₖ (k filas, n columnas)")
        self.c_in_cl = VectorInputTable("Coeficientes c (k valores)", 2)
        run_btn = btn("Calcular Combinación")
        run_btn.clicked.connect(self.on_run_comb_lineal)
        layout.addLayout(dims)
        layout.addWidget(self.V_in_cl)
        layout.addWidget(self.c_in_cl)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        self.k_in_cl.edit.textChanged.connect(self._sync_cl)
        self.n_in_cl.edit.textChanged.connect(self._sync_cl)
        self._sync_cl()
        return page

    def _sync_cl(self):
        try:
            k, n = int(self.k_in_cl.text()), int(self.n_in_cl.text())
            self.V_in_cl.set_size(k, n)
            self.c_in_cl.set_size(k)
        except ValueError: pass

    def on_run_comb_lineal(self):
        try:
            V = self.V_in_cl.to_matrix()
            coef = self.c_in_cl.to_vector()
            out = combinacion_lineal_explicada(V, coef, dec=4)
            self._set_text(f"{out['texto']}\n\nComo lista: {out['resultado']}")
        except Exception as e:
            self._set_text(f"Error: {e}")

    # --- Página 4: Ecuación Vectorial ---
    def _create_ecu_vec_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        dims = QtWidgets.QHBoxLayout()
        self.k_in_ev = LabeledEdit("k (n° vectores):", "2", default_value="2")
        self.n_in_ev = LabeledEdit("n (dimensión):", "3", default_value="3")
        dims.addWidget(self.k_in_ev)
        dims.addWidget(self.n_in_ev)
        self.V_in_ev = MatrixTable(2, 3, "Lista de vectores v₁..vₖ (k filas, n columnas)")
        self.b_in_ev = VectorInputTable("Vector b (n valores)", 3)
        run_btn = btn("Resolver Ecuación Vectorial")
        run_btn.clicked.connect(self.on_run_ecu_vec)
        layout.addLayout(dims)
        layout.addWidget(self.V_in_ev)
        layout.addWidget(self.b_in_ev)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        self.k_in_ev.edit.textChanged.connect(self._sync_ev)
        self.n_in_ev.edit.textChanged.connect(self._sync_ev)
        self._sync_ev()
        return page

    def _sync_ev(self):
        try:
            k, n = int(self.k_in_ev.text()), int(self.n_in_ev.text())
            self.V_in_ev.set_size(k, n)
            self.b_in_ev.set_size(n)
        except ValueError: pass

    def on_run_ecu_vec(self):
        try:
            V = self.V_in_ev.to_matrix()
            bvals = self.b_in_ev.to_vector()
            out = ecuacion_vectorial(V, bvals)
            lines = [p + "\n" for p in out.get("reportes", [])]
            lines.append(f"Estado: {out.get('tipo') or out.get('estado')}\n")
            lines.append(formatear_solucion_parametrica(out, dec=4, fracciones=True))
            self._set_text("\n".join(lines))
        except Exception as e:
            self._set_text(f"Error: {e}")

# =========================
#   Programa 5 (Matrices) — con cálculo elemento a elemento
# =========================
class TabProg5(QtWidgets.QWidget):
    """Programa 5 — Operaciones con matrices y verificación de propiedades de la traspuesta."""
    def __init__(self, parent=None):
        super().__init__(parent)
        main = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.sp_m = QtWidgets.QSpinBox(); self.sp_m.setRange(1, 50); self.sp_m.setValue(2)
        self.sp_m.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons); _disable_spin_wheel(self.sp_m)
        self.sp_n = QtWidgets.QSpinBox(); self.sp_n.setRange(1, 50); self.sp_n.setValue(2)
        self.sp_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons); _disable_spin_wheel(self.sp_n)
        top.addWidget(QtWidgets.QLabel("A: m=")); top.addWidget(self.sp_m)
        top.addWidget(QtWidgets.QLabel(" n=")); top.addWidget(self.sp_n)
        top.addSpacing(16)
        self.sp_r = QtWidgets.QSpinBox(); self.sp_r.setRange(1, 50); self.sp_r.setValue(2)
        self.sp_r.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons); _disable_spin_wheel(self.sp_r)
        self.sp_p = QtWidgets.QSpinBox(); self.sp_p.setRange(1, 50); self.sp_p.setValue(2)
        self.sp_p.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons); _disable_spin_wheel(self.sp_p)
        top.addWidget(QtWidgets.QLabel("B: m=")); top.addWidget(self.sp_r)
        top.addWidget(QtWidgets.QLabel(" n=")); top.addWidget(self.sp_p)
        top.addSpacing(16)
        self.k_edit = QtWidgets.QLineEdit("1")
        self.k_edit.setValidator(QtGui.QDoubleValidator())
        self.k_edit.setMaximumWidth(120)
        top.addWidget(QtWidgets.QLabel("r =")); top.addWidget(self.k_edit)
        top.addStretch(1)
        main.addLayout(top)

        center = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0,0,0,0)
        self.tblA = MatrixTable(2,2,"Matriz A")
        self.tblB = MatrixTable(2,2,"Matriz B")
        btnsAB = QtWidgets.QHBoxLayout()
        self.btn_fill = btn("Rellenar ceros")
        self.btn_clear = btn("Limpiar")
        btnsAB.addWidget(self.btn_fill); btnsAB.addWidget(self.btn_clear); btnsAB.addStretch(1)
        ll.addWidget(self.tblA); ll.addWidget(self.tblB); ll.addLayout(btnsAB)

        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(6,0,0,0)
        ops_layout = QtWidgets.QGridLayout()
        self.btn_sum = btn("A + B")
        self.btn_res = btn("A - B")
        self.btn_kA  = btn("r · A")
        self.btn_AB  = btn("A · B")
        self.btn_AT  = btn("A^T")
        self.btn_ATT = btn("Verificar (A^T)^T = A")
        self.btn_rSumT = btn("Verificar (r(A+B))^T")
        ops_layout.addWidget(self.btn_sum, 0, 0)
        ops_layout.addWidget(self.btn_res, 0, 1)
        ops_layout.addWidget(self.btn_kA, 0, 2)
        ops_layout.addWidget(self.btn_AB, 0, 3)
        ops_layout.addWidget(self.btn_AT, 1, 0)
        ops_layout.addWidget(self.btn_ATT, 1, 1)
        ops_layout.addWidget(self.btn_rSumT, 1, 2, 1, 2)
        rl.addLayout(ops_layout)

        self.out = OutputArea()
        rl.addWidget(self.out)

        # === Botón global de formato (pestaña) ===
        self._show_frac = False
        self._last_dec = ""
        self._last_frac = ""
        self.btn_fmt = btn("Cambiar a fracciones")
        rl.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_fmt.clicked.connect(self._toggle_fmt)

        center.addWidget(left)
        center.addWidget(right)
        center.setStretchFactor(0, 1)
        center.setStretchFactor(1, 1)
        main.addWidget(center)

        self.sp_m.valueChanged.connect(self._sync_A)
        self.sp_n.valueChanged.connect(self._sync_A)
        self.sp_r.valueChanged.connect(self._sync_B)
        self.sp_p.valueChanged.connect(self._sync_B)

        self.btn_fill.clicked.connect(self._fill_zeros)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_sum.clicked.connect(self.on_sum)
        self.btn_res.clicked.connect(self.on_res)
        self.btn_kA.clicked.connect(self.on_kA)
        self.btn_AB.clicked.connect(self.on_AB)
        self.btn_AT.clicked.connect(self.on_AT)
        self.btn_ATT.clicked.connect(self.on_ATT_prop)
        self.btn_rSumT.clicked.connect(self.on_rSumT_prop)

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self.on_sum)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self.on_sum)

        self.sp_n.valueChanged.connect(lambda v: self.sp_r.setValue(v))
        self._sync_A(); self._sync_B()

    # helpers formato
    def _set_text(self, base_text: str):
        self._last_dec = text_to_decimals(base_text)
        self._last_frac = text_to_fractions(base_text)
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        self.out.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def _sync_A(self):
        self.tblA.set_size(self.sp_m.value(), self.sp_n.value())

    def _sync_B(self):
        self.tblB.set_size(self.sp_r.value(), self.sp_p.value())

    def _fill_zeros(self):
        self.tblA.fill_zeros(); self.tblB.fill_zeros()

    def _clear_all(self):
        self.tblA.clear_all(); self.tblB.clear_all(); self._set_text("")

    def _k_value(self) -> float:
        t = self.k_edit.text().strip() or "0"
        return float(evaluar_expresion(t, exacto=False))

    def _print(self, lines: list[str]):
        self._set_text("\n".join(lines))

    def on_sum(self):
        try:
            A = self.tblA.to_matrix()
            B = self.tblB.to_matrix()
            out = suma_matrices_explicada(A, B)
            lines = ["--- Pasos ---"] + out["pasos"] + ["", "Resultado C = A + B:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(A + B)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "A^T:", format_matrix_text(out["AT"]), "", "B^T:", format_matrix_text(out["BT"])]
            lines += ["", "A^T + B^T:", format_matrix_text(out["AT_mas_BT"]), "", ">>> " + out["conclusion"]]
            self._print(lines)
        except Exception as e:
            self._print(["Error:", str(e)])

    def on_res(self):
        try:
            A = self.tblA.to_matrix()
            B = self.tblB.to_matrix()
            out = resta_matrices_explicada(A, B)
            lines = ["--- Pasos ---"] + out["pasos"] + ["", "Resultado C = A - B:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(A - B)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "A^T:", format_matrix_text(out["AT"]), "", "B^T:", format_matrix_text(out["BT"])]
            lines += ["", "A^T - B^T:", format_matrix_text(out["AT_menos_BT"]), "", ">>> " + out["conclusion"]]
            self._print(lines)
        except Exception as e:
            self._print(["Error:", str(e)])

    def on_kA(self):
        try:
            A = self.tblA.to_matrix()
            r = self._k_value()
            out = producto_escalar_explicado(r, A)
            lines = ["--- Pasos ---"] + out["pasos"] + ["", "Resultado r·A:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(rA)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "A^T:", format_matrix_text(out["AT"]), "", "r·A^T:", format_matrix_text(out["kAT"])]
            lines += ["", ">>> " + out["conclusion"]]
            self._print(lines)
        except Exception as e:
            self._print(["Error:", str(e)])

    def on_AB(self):
        try:
            if self.sp_r.value() != self.sp_n.value():
                raise ValueError("Para A·B se requiere filas(B) = columnas(A). Ajusta los spinners.")
            A = self.tblA.to_matrix()
            B = self.tblB.to_matrix()
            out = producto_matrices_explicado(A, B)
            lines = ["--- Pasos ---"] + out["pasos"] + ["", "Resultado C = A·B:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(AB)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "B^T:", format_matrix_text(out["BT"]), "", "A^T:", format_matrix_text(out["AT"])]
            lines += ["", "B^T·A^T:", format_matrix_text(out["BT_por_AT"]), "", ">>> " + out["conclusion"]]
            self._print(lines)
        except Exception as e:
            self._print(["Error:", str(e)])

    def on_AT(self):
        try:
            A = self.tblA.to_matrix()
            out = traspuesta_explicada(A)
            lines = ["--- Pasos ---", "Se intercambian filas por columnas."]
            lines += ["", "Resultado A^T:", format_matrix_text(out["resultado"])]
            self._print(lines)
        except Exception as e:
            self._print(["Error:", str(e)])

    def on_ATT_prop(self):
        try:
            A = self.tblA.to_matrix()
            out = traspuesta_explicada(A)
            lines = ["--- Verificación de (A^T)^T = A ---"]
            lines += out["pasos"]
            lines += [
                "", "A^T:", format_matrix_text(out["resultado"]),
                "", "(A^T)^T:", format_matrix_text(out["ATT"]),
                "", ">>> " + out["conclusion"]
            ]
            self._print(lines)
        except Exception as e:
            self._print(["Error:", str(e)])

    def on_rSumT_prop(self):
        try:
            A = self.tblA.to_matrix()
            B = self.tblB.to_matrix()
            r = self._k_value()
            out = propiedad_r_suma_traspuesta_explicada(A, B, r)
            lines = ["--- Verificación de (r(A+B))^T = r(A^T+B^T) ---"]
            lines += out["pasos"]
            lines += [
                "", "Lado Izquierdo (r(A+B))^T:", format_matrix_text(out["izquierda"]),
                "", "Lado Derecho r(A^T+B^T):", format_matrix_text(out["derecha"]),
                "", ">>> " + out["conclusion"]
            ]
            self._print(lines)
        except Exception as e:
            self._print(["Error:", str(e)])

# =========================
#   Programa 6 (Inversa: Gauss–Jordan + propiedades) — botón global por pestaña
# =========================
class TabProg6(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)

        # --- fila superior ---
        top = QtWidgets.QHBoxLayout()
        self.sp_n = QtWidgets.QSpinBox(); self.sp_n.setRange(1, 50); self.sp_n.setValue(2)
        self.sp_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        _disable_spin_wheel(self.sp_n)
        top.addWidget(QtWidgets.QLabel("n:")); top.addWidget(self.sp_n)
        top.addStretch(1)
        root.addLayout(top)

        # --- centro: tabla (izq) + conclusiones (der) ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0,0,0,0)
        self.tblA = MatrixTable(2, 2, "Matriz A (n×n)")
        ll.addWidget(self.tblA)

        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(6,0,0,0)
        self.summary = OutputArea()
        rl.addWidget(self.summary)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter)

        # --- barra de acciones ---
        actions = QtWidgets.QHBoxLayout()
        self.btn_inv   = btn("Calcular A^{-1} y |A|")
        self.btn_props = btn("Verificar propiedades (c)(d)(e)")
        self.btn_all   = btn("Todo junto")
        self.btn_clear = btn("Limpiar")
        actions.addStretch(1)
        for b in (self.btn_inv, self.btn_props, self.btn_all, self.btn_clear):
            actions.addWidget(b)
        actions.addStretch(1)
        root.addLayout(actions)

        # --- panel inferior: TODO el procedimiento Gauss–Jordan ---
        self.steps_all = OutputArea()
        self.steps_all.setPlaceholderText("Aquí se mostrará TODO el procedimiento de Gauss–Jordan sobre [A | I].")
        root.addWidget(self.steps_all)

        # === Botón global de formato (pestaña) — controla summary y steps ===
        self._show_frac = False
        self._sum_dec = ""; self._sum_frac = ""
        self._steps_dec = ""; self._steps_frac = ""
        self.btn_fmt = btn("Cambiar a fracciones")
        root.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_fmt.clicked.connect(self._toggle_fmt)

        # conexiones / atajos
        self.sp_n.valueChanged.connect(self._sync_n)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_inv.clicked.connect(self.on_inv)
        self.btn_props.clicked.connect(self.on_props)
        self.btn_all.clicked.connect(self.on_all)

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+I"), self, activated=self.on_inv)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+P"), self, activated=self.on_props)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self, activated=self.on_all)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"),   self, activated=self._clear_all)

        self._sync_n()

    # ---------- utilidades ----------
    def _sync_n(self):
        self.tblA.set_size(self.sp_n.value(), self.sp_n.value())

    def _clear_all(self):
        self.tblA.clear_all()
        self._set_summary("")
        self._set_steps("")

    def _A(self):
        n = self.sp_n.value()
        A = self.tblA.to_matrix()
        if len(A) != n or len(A[0]) != n:
            raise ValueError("A debe ser cuadrada n×n.")
        return A

    # helpers formato
    def _set_summary(self, base_text: str):
        self._sum_dec  = text_to_decimals(base_text)
        self._sum_frac = text_to_fractions(base_text)
        self.summary.clear_and_write(self._sum_frac if self._show_frac else self._sum_dec)

    def _set_steps(self, base_text: str):
        self._steps_dec  = text_to_decimals(base_text)
        self._steps_frac = text_to_fractions(base_text)
        self.steps_all.clear_and_write(self._steps_frac if self._show_frac else self._steps_dec)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        self.summary.clear_and_write(self._sum_frac if self._show_frac else self._sum_dec)
        self.steps_all.clear_and_write(self._steps_frac if self._show_frac else self._steps_dec)

    # ---------- acciones ----------
    def on_inv(self):
        try:
            A = self._A()
            inv = inversa_por_gauss_jordan(A, dec=4)

            lines = []
            lines += ["=== Inversa por Gauss–Jordan ===", ""]
            lines += ["Determinante:", inv.get("det_texto","|A| = (desconocido)"), ""]
            lines += ["Conclusión:", inv.get("conclusion","")]
            if inv.get("estado") == "ok":
                lines += ["", "A^{-1} =", format_matrix_text(inv["Ainv"])]
            self._set_summary("\n".join(lines))

            pasos = inv.get("pasos", [])
            self._set_steps("\n\n".join(pasos) if pasos else "(No se generaron pasos.)")

        except Exception as e:
            self._set_summary("Error: " + str(e))
            self._set_steps("")

    def on_props(self):
        try:
            A = self._A()
            props = verificar_propiedades_invertibilidad(A, dec=4)

            lines = []
            lines += ["=== Propiedades (c)(d)(e) ===", ""]
            lines += [f"Pivotes (1-index): {props.get('pivotes', 'N/A')}",
                      f"Rango: {props.get('rango', 'N/A')}", ""]
            lines += [props.get("explicacion", "No se pudo generar la explicación.")]
            
            if props.get("detalle_sistema_homogeneo"):
                lines.append("\n--- Detalles del sistema homogéneo Ax=0 ---")
                lines.append(formatear_solucion_parametrica(props["detalle_sistema_homogeneo"]))
                
            self._set_summary("\n".join(lines))
            self._set_steps("(Ejecuta 'Calcular A^{-1} y |A|' o 'Todo junto' para ver el procedimiento).")

        except Exception as e:
            self._set_summary("Error: " + str(e))
            self._set_steps("")

    def on_all(self):
        try:
            A = self._A()
            full = programa_inversa_con_propiedades(A, dec=4)
            inv, props = full["inversa"], full["propiedades"]

            lines = []
            lines += ["=== Inversa + Propiedades ===", ""]
            lines += ["Determinante:", inv.get("det_texto","|A| = (desconocido)"), ""]
            lines += ["Conclusión (inversa):", inv.get("conclusion","")]
            if inv.get("estado") == "ok":
                lines += ["", "A^{-1} =", format_matrix_text(inv["Ainv"])]
            lines += ["", "--- Propiedades (c)(d)(e) ---", props.get("explicacion",""), ""]
            lines += ["Conclusión global:", full.get("conclusion_global","")]
            self._set_summary("\n".join(lines))

            pasos = inv.get("pasos", [])
            self._set_steps("\n\n".join(pasos) if pasos else "(No se generaron pasos.)")

        except Exception as e:
            self._set_summary("Error: " + str(e))
            self._set_steps("")

# =========================
#   Ventana principal
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Álgebra Lineal — GUI (PyQt6)")
        self.resize(1200, 820)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(TabGaussJordan(), "Gauss-Jordan (Ab)")
        tabs.addTab(TabAXeqB(), "AX=B")
        tabs.addTab(TabProg4(), "Dependencia")         # 👈 renombrado
        tabs.addTab(TabVectores(), "Vectores")
        tabs.addTab(TabProg5(), "Operaciones con Matrices")
        tabs.addTab(TabProg6(), "Inversa")   

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Listo")

        # Oculta las flechas de TODOS los SpinBox del árbol y desactiva la rueda
        hide_all_spin_buttons(self)

# =========================
#   main() con tema claro
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_green_theme(app)  # 👈 aplica el look & feel verde

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
