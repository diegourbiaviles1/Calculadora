# widgets.py
from __future__ import annotations
from typing import List
import re
from fractions import Fraction
import math

from PyQt6 import QtWidgets, QtCore, QtGui
from utilidad import evaluar_expresion, fmt_number, DEFAULT_DEC, DEFAULT_EPS


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

def _safe_float(x: str) -> float:
    x = (x or "").strip()
    if not x:
        return 0.0
    return float(evaluar_expresion(x, exacto=False))

# =========================
#   Widgets básicos y utilidades
# =========================
def mono_font():
    f = QtGui.QFont("Consolas")
    f.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
    f.setPointSize(10)
    return f


def _disable_spin_wheel(sb: QtWidgets.QAbstractSpinBox):
    def _no_wheel(event): 
        event.ignore()
    sb.wheelEvent = _no_wheel # type: ignore


def hide_all_spin_buttons(root: QtWidgets.QWidget):
    """
    Oculta los botones ↑↓ de TODOS los spin boxes del árbol y desactiva la rueda.
    """
    for sb in root.findChildren(QtWidgets.QAbstractSpinBox):
        sb.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        _disable_spin_wheel(sb)
# ------------------------------------------------------------------ 

def btn(text: str, kind: str = "primary"):
    b = QtWidgets.QPushButton(text)
    b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
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

class OutputArea(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(mono_font())
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def clear_and_write(self, s: str):
        self.clear(); self.setPlainText(s or "")

    def _set_text(self, base_text: str):
        self._last_dec = text_to_decimals(base_text)
        self._last_frac = text_to_fractions(base_text)
        # Esto es para que las pestañas que usen un solo OutputArea puedan cambiar el formato.
        # En las pestañas Tab... que solo usan un OutputArea, este método debe ser llamado desde un botón global de la pestaña.
        self._show_frac = getattr(self, '_show_frac', False) # Inicializa si no existe
        self.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

    def set_output_format(self, base_text: str, show_frac: bool):
        self._last_dec = text_to_decimals(base_text)
        self._last_frac = text_to_fractions(base_text)
        self._show_frac = show_frac
        self.clear_and_write(self._last_frac if self._show_frac else self._last_dec)

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

class MatrixAugTable(QtWidgets.QWidget):
    """
    Editor para matrices aumentadas (m x (n+1)).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)

        # Barra superior: m, n
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("m:"))
        self.spin_m = QtWidgets.QSpinBox(); self.spin_m.setRange(1, 999); self.spin_m.setValue(3)
        _disable_spin_wheel(self.spin_m)
        top.addWidget(self.spin_m)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("n:"))
        self.spin_n = QtWidgets.QSpinBox(); self.spin_n.setRange(1, 999); self.spin_n.setValue(3)
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

    def to_augmented(self) -> list[list[float]]:
        m = self.table.rowCount()
        n = self.table.columnCount() - 1
        Ab = []
        for i in range(m):
            fila = []
            for j in range(n + 1):
                item = self.table.item(i, j)
                txt = "" if item is None else item.text()
                fila.append(_safe_float(txt))
            Ab.append(fila)
        return Ab