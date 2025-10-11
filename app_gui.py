# app_gui.py — GUI PyQt6 completa para tu proyecto
from __future__ import annotations
from typing import List
import sys

from PyQt6 import QtWidgets, QtCore, QtGui

from utilidad import evaluar_expresion, fmt_number, DEFAULT_DEC
from sistema_lineal import SistemaLineal, formatear_solucion_parametrica
from homogeneo import (
    resolver_sistema_homogeneo_y_no_homogeneo,
    resolver_dependencia_lineal_con_homogeneo,
)
from algebra_vector import (
    resolver_AX_igual_B,
    multiplicacion_matriz_vector_explicada,  # por si lo quieres usar luego
    sistema_a_forma_matricial,               # por si lo quieres usar luego
    ecuacion_vectorial,
    combinacion_lineal_explicada,
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
    # n variables + 1 término independiente
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
    """
    Devuelve un bloque de texto con la matriz aumentada centrada y un solo par de corchetes grandes.
    Usa ⎡ ⎤ / ⎢ ⎥ / ⎣ ⎦ y alinea columnas según el ancho máximo de cada una.
    """
    m = table.table.rowCount()
    n = table.table.columnCount() - 1
    # Leer como cadenas
    grid = []
    for i in range(m):
        row = []
        for j in range(n + 1):
            it = table.table.item(i, j)
            row.append((it.text() if it else "0").strip())
        grid.append(row)

    # Ancho máximo por columna (incluye b)
    widths = [0] * (n + 1)
    for j in range(n + 1):
        widths[j] = max(len(grid[i][j]) if grid[i][j] else 1 for i in range(m))

    # Construir filas con padding; separador antes de b
    rows_txt = []
    for i in range(m):
        left = "  ".join(grid[i][j].rjust(widths[j]) for j in range(n))
        right = grid[i][n].rjust(widths[n])
        rows_txt.append(f"{left}  |  {right}" if n > 0 else right)

    # Corchetes grandes
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

def btn(text):
    b = QtWidgets.QPushButton(text)
    b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    return b

class LabeledEdit(QtWidgets.QWidget):
    def __init__(self, label, placeholder="", parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.lbl = QtWidgets.QLabel(label)
        self.edit = QtWidgets.QLineEdit(); self.edit.setPlaceholderText(placeholder)
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
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # << centrar texto


    def clear_and_write(self, s: str):
        self.clear(); self.setPlainText(s or "")

# =========================
#   Editor tabular Ab
# =========================
class MatrixAugTable(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)

        # Barra superior: m, n (sin botones)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("m:"))
        self.spin_m = QtWidgets.QSpinBox(); self.spin_m.setRange(1, 999); self.spin_m.setValue(3)
        top.addWidget(self.spin_m)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("n:"))
        self.spin_n = QtWidgets.QSpinBox(); self.spin_n.setRange(1, 999); self.spin_n.setValue(3)
        top.addWidget(self.spin_n)
        top.addStretch(1)
        lay.addLayout(top)

        # Tabla
        self.table = QtWidgets.QTableWidget(3, 7)  # por defecto 3x(7), ajustado por el tamaño de la matriz
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Desactivar encabezados verticales (números a la izquierda)
        self.table.verticalHeader().setVisible(False)

        # Aquí es donde modificamos los encabezados
        self._refresh_headers()

        lay.addWidget(self.table)

        # Conexiones
        self.spin_m.valueChanged.connect(self._resize_table)
        self.spin_n.valueChanged.connect(self._resize_table)

    def _refresh_headers(self):
        n = self.table.columnCount() - 1  # No incluye la columna de términos independientes
        labels = [f"X{j+1}" for j in range(n)]  # Solo X1, X2, X3, ..., Xn
        labels.append("B")  # Añadir "B" como encabezado para la columna de términos independientes
        self.table.setHorizontalHeaderLabels(labels)


    def _resize_table(self):
        m = int(self.spin_m.value())
        n = int(self.spin_n.value())
        if self.table.rowCount() != m:
            self.table.setRowCount(m)
        if self.table.columnCount() != (n + 1):  # Para incluir una columna extra por el término independiente
            self.table.setColumnCount(n + 1)
            self._refresh_headers()

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
        """Limpia todas las celdas de la tabla."""
        m, n = self.table.rowCount(), self.table.columnCount()
        for i in range(m):
            for j in range(n):
                item = self.table.item(i, j)
                if item is None:
                    item = QtWidgets.QTableWidgetItem("")
                    self.table.setItem(i, j, item)
                else:
                    item.setText("")  # Borrar el contenido de la celda








# =========================
#   Pestañas de la app
# =========================
class TabGaussJordan(QtWidgets.QWidget):
    """Resolver por Gauss–Jordan a partir de matriz aumentada (tabla a la izquierda, vista inicial a la derecha)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)

        # --- Grupo superior: mitad tabla / mitad vista ---
        self.group = QtWidgets.QGroupBox("Tamaño de la matriz")
        gl = QtWidgets.QVBoxLayout(self.group)

        # Split horizontal: izquierda (tabla), derecha (vista inicial)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # izquierda: editor de Ab
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left); left_lay.setContentsMargins(0,0,0,0)
        self.aug_table = MatrixAugTable()
        left_lay.addWidget(self.aug_table)

        # derecha: vista de la matriz aumentada inicial (solo lectura)
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right); right_lay.setContentsMargins(6,0,0,0)
        self.lbl_preview = QtWidgets.QLabel("Matriz aumentada inicial")
        self.lbl_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)  # << centrar título

        self.preview = OutputArea()
        self.preview.setReadOnly(True)
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)       # << centrar texto

        self.preview.setReadOnly(True)
        right_lay.addWidget(self.lbl_preview)
        right_lay.addWidget(self.preview)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)  # mitad
        splitter.setStretchFactor(1, 1)  # mitad
        gl.addWidget(splitter)

        # --- Barra de botones al pie del grupo (centrada) ---
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

        mlay.addWidget(self.group)

        # --- (Opcional) caja de texto debajo por si quieres pegar matrices ---
        self.lbl_procedimiento = QtWidgets.QLabel("Procedimiento (reducción por filas)")
        self.lbl_procedimiento.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        mlay.addWidget(self.lbl_procedimiento)


        # --- Salida de pasos/resultados ---
        self.out = OutputArea()
        mlay.addWidget(self.out)

        # conexiones
        self.btn_zeros.clicked.connect(self.aug_table.fill_zeros)
        self.btn_clear.clicked.connect(self.aug_table.clear_all)
        self.btn_solve.clicked.connect(self.on_run_table)

        # Actualizar vista de la derecha cuando cambie algo
        self.aug_table.table.itemChanged.connect(self.update_preview)
        self.aug_table.spin_m.valueChanged.connect(lambda _: self.update_preview())
        self.aug_table.spin_n.valueChanged.connect(lambda _: self.update_preview())

        # Atajos
        sc1 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
        sc1.activated.connect(self.on_run_table)
        sc2 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self)
        sc2.activated.connect(self.on_run_table)

        # primera vista
        self.update_preview()

    # ---- utilidades ----
    def update_preview(self):
        self.preview.clear_and_write(pretty_augmented_from_table(self.aug_table))


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
        lines.append(formatear_solucion_parametrica(out, nombres_vars=None, dec=4, fracciones=True))
        self.out.clear_and_write("\n".join(lines))

    # ---- acción principal ----
    def on_run_table(self):
        try:
            Ab = table_to_augmented(self.aug_table)
            sl = SistemaLineal(Ab, decimales=4)
            out = sl.gauss_jordan()
            self._render_out(out)
        except Exception as e:
            self.out.clear_and_write("Error: " + str(e))


class TabProg4(QtWidgets.QWidget):
    """Programa 4: Ax=b (pasos+paramétrica) y dependencia A·c=0."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)

        dims = QtWidgets.QHBoxLayout()
        self.m_in = LabeledEdit("m:", "filas"); self.n_in = LabeledEdit("n:", "columnas")
        self.btn_export = btn("Exportar pasos…")
        dims.addWidget(self.m_in); dims.addWidget(self.n_in); dims.addStretch(1); dims.addWidget(self.btn_export)
        mlay.addLayout(dims)

        self.A = MatrixInput("Matriz A (m x n)")
        self.b = MatrixInput("Vector b (m valores, misma línea o una por línea)", height=70)
        mlay.addWidget(self.A); mlay.addWidget(self.b)

        self.run = btn("Resolver Ax=b + Dependencia (A·c=0)")
        mlay.addWidget(self.run)

        self.out = OutputArea(); mlay.addWidget(self.out)
        self.run.clicked.connect(self.on_run)
        self.btn_export.clicked.connect(lambda: save_text_to_file(self, self.out.toPlainText(), "programa4.txt"))

    def on_run(self):
        try:
            m = int(self.m_in.text()); n = int(self.n_in.text())
            A = parse_matrix(self.A.text(), m, n)
            b_vals = parse_nums(self.b.text().replace("\n", " "))
            if len(b_vals) != m:
                raise ValueError(f"b debe tener {m} valores.")
            info = resolver_sistema_homogeneo_y_no_homogeneo(A, b_vals)

            lines = []
            lines.append("=== PASOS (Gauss-Jordan aplicado a Ax = b) ===")
            for p in info.get("pasos", []): lines.append(p)
            lines.append("\n=== SOLUCIÓN GENERAL (forma paramétrica) ===")
            lines.append(info["salida_parametrica"])
            lines.append("\n=== CONCLUSIÓN DEL SISTEMA ===")
            lines.append(info["conclusion"])

            dep = resolver_dependencia_lineal_con_homogeneo(A)
            lines.append("\n\n=== ANÁLISIS DE DEPENDENCIA LINEAL ===")
            lines.append(dep.get("dependencia", "(sin análisis)"))
            lines.append("\n=== PASOS (Gauss-Jordan aplicado a A·c = 0) ===")
            for p in dep.get("pasos", []): lines.append(p)
            lines.append("\n=== COMBINACIÓN LINEAL (forma paramétrica de los coeficientes c) ===")
            lines.append(dep.get("salida_parametrica", ""))

            self.out.clear_and_write("\n".join(lines))
        except Exception as e:
            self.out.clear_and_write("Error: " + str(e))

class TabAXeqB(QtWidgets.QWidget):
    """Resolver AX=B (B vector o matriz)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)

        # --- Fila superior: m n ---
        dims = QtWidgets.QHBoxLayout()
        self.m_in = LabeledEdit("m:", "filas")
        self.n_in = LabeledEdit("n:", "columnas")
        dims.addWidget(self.m_in); dims.addWidget(self.n_in)
        mlay.addLayout(dims)

        # --- Entrada A (m x n) ---
        self.A = MatrixAugTable()  # Esta tabla es donde se define la matriz A
        mlay.addWidget(self.A)

        # --- Entrada b como vector ---
        self.b = MatrixInput("Vector b (m valores en una sola línea)")
        self.b.txt.setPlaceholderText("Ej.: 1 2 3")
        mlay.addWidget(self.b)

        # --- Mostrar el vector X (dependiendo de los coeficientes) ---
        self.lbl_vector = QtWidgets.QLabel("X = ")
        self.vector_x = QtWidgets.QLabel("")  # Esta etiqueta se actualizará para mostrar X = [x1, x2, ..., xn]
        mlay.addWidget(self.lbl_vector)
        mlay.addWidget(self.vector_x)

        # --- Botón de resolver ---
        self.run = btn("Resolver AX=B")
        mlay.addWidget(self.run)

        # --- Salida ---
        self.out = OutputArea(); mlay.addWidget(self.out)
        self.run.clicked.connect(self.on_run)

        # conexiones de actualización
        self.m_in.edit.textChanged.connect(self._sync_dims_to_table)
        self.n_in.edit.textChanged.connect(self._sync_dims_to_table)

        # Atajos
        sc1 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
        sc1.activated.connect(self.on_run)
        sc2 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self)
        sc2.activated.connect(self.on_run)

        self.update_table()

    def _sync_dims_to_table(self):
        try:
            m = int(self.m_in.text())
            n = int(self.n_in.text())
            self.A.set_size(m, n)
            self.update_table()
        except Exception:
            pass

    def update_table(self):
        self.A._refresh_headers()  # Actualizamos los encabezados a "A"
        self.A.fill_zeros()  # Opcional, si quieres rellenar con ceros por defecto
        self.update_vector_x()

    def update_vector_x(self):
        # Generar el vector X con coeficientes X1, X2, ..., Xn
        n = int(self.n_in.text())
        vector_str = "[" + "  ".join([f"x{j+1}" for j in range(n)]) + "]"
        self.vector_x.setText(vector_str)

    def update_vector_x(self):
    # Generar el vector X con coeficientes X1, X2, ..., Xn
        n = int(self.n_in.text())
        vector_str = "[" + "  ".join([f"x{j+1}" for j in range(n)]) + "]"
        self.vector_x.setText(vector_str)

    
    def on_run(self):
        try:
            m = int(self.m_in.text()); n = int(self.n_in.text())
            A = parse_matrix(self.A.to_text(), m, n)
            b_vals = parse_vector(self.b.text(), m)
            out = resolver_AX_igual_B(A, b_vals)
            lines = []
            for p in out.get("reportes", []): lines.append(p)
            if out.get("estado") == "ok":
                if "x" in out:
                    lines.append("\nx = " + str(out["x"]))
                if "X" in out:
                    lines.append("\nX =")
                    for fila in out["X"]:
                        lines.append("  " + "  ".join(fmt_number(x, DEFAULT_DEC, False) for x in fila))
            else:
                lines.append("\nEstado: " + str(out.get("estado")))
            self.out.clear_and_write("\n".join(lines))
        except Exception as e:
            self.out.clear_and_write("Error: " + str(e))



    def _sync_dims_to_table(self):
        try:
            m = int(self.m_in.text())
            n = int(self.n_in.text())
            self.A.set_size(m, n)
            self.update_table()
        except Exception:
            pass

    def update_table(self):
        self.A._refresh_headers()
        self.A.fill_zeros()
        self.update_vector_x()

    def update_vector_x(self):
        # Generar el vector X con coeficientes X1, X2, ..., Xn
        n = int(self.n_in.text())
        vector_str = "[" + "  ".join([f"x{j+1}" for j in range(n)]) + "]"
        self.vector_x.setText(vector_str)
    
    def on_run(self):
        try:
            m = int(self.m_in.text()); n = int(self.n_in.text())
            A = parse_matrix(self.A.to_text(), m, n)
            b_vals = parse_vector(self.b.text(), m)
            out = resolver_AX_igual_B(A, b_vals)
            lines = []
            for p in out.get("reportes", []): lines.append(p)
            if out.get("estado") == "ok":
                if "x" in out:
                    lines.append("\nx = " + str(out["x"]))
                if "X" in out:
                    lines.append("\nX =")
                    for fila in out["X"]:
                        lines.append("  " + "  ".join(fmt_number(x, DEFAULT_DEC, False) for x in fila))
            else:
                lines.append("\nEstado: " + str(out.get("estado")))
            self.out.clear_and_write("\n".join(lines))
        except Exception as e:
            self.out.clear_and_write("Error: " + str(e))


    def _sync_dims_to_table(self):
        try:
            m = int(self.m_in.text())
            n = int(self.n_in.text())
            self.A.set_size(m, n)
            self.update_table()
        except Exception:
            pass

    def update_table(self):
        self.A._refresh_headers()
        self.A.fill_zeros()
    
    def on_run(self):
        try:
            m = int(self.m_in.text()); n = int(self.n_in.text())
            A = parse_matrix(self.A.to_text(), m, n)
            b_vals = parse_vector(self.b.text(), m)
            out = resolver_AX_igual_B(A, b_vals)
            lines = []
            for p in out.get("reportes", []): lines.append(p)
            if out.get("estado") == "ok":
                if "x" in out:
                    lines.append("\nx = " + str(out["x"]))
                if "X" in out:
                    lines.append("\nX =")
                    for fila in out["X"]:
                        lines.append("  " + "  ".join(fmt_number(x, DEFAULT_DEC, False) for x in fila))
            else:
                lines.append("\nEstado: " + str(out.get("estado")))
            self.out.clear_and_write("\n".join(lines))
        except Exception as e:
            self.out.clear_and_write("Error: " + str(e))

class TabVectores(QtWidgets.QWidget):
    """Herramientas: ecuación vectorial y combinación lineal."""
    def __init__(self, parent=None):
        super().__init__(parent)
        mlay = QtWidgets.QVBoxLayout(self)

        # Ecuación vectorial
        group1 = QtWidgets.QGroupBox("Ecuación vectorial  (¿b en span{v1..vk}?)")
        g1 = QtWidgets.QVBoxLayout(group1)
        dims1 = QtWidgets.QHBoxLayout()
        self.k1 = LabeledEdit("k:", "n° vectores")
        self.n1 = LabeledEdit("n:", "dimensión")
        dims1.addWidget(self.k1); dims1.addWidget(self.n1)
        g1.addLayout(dims1)
        self.V1 = MatrixInput("Lista de vectores (uno por línea, n valores)")
        self.b1 = MatrixInput("b (n valores en una línea o multilinea)", height=60)
        self.btn1 = btn("Resolver ecuación vectorial")
        g1.addWidget(self.V1); g1.addWidget(self.b1); g1.addWidget(self.btn1)
        mlay.addWidget(group1)

        # Combinación lineal
        group2 = QtWidgets.QGroupBox("Combinación lineal de vectores")
        g2 = QtWidgets.QVBoxLayout(group2)
        dims2 = QtWidgets.QHBoxLayout()
        self.k2 = LabeledEdit("k:", "n° vectores")
        self.n2 = LabeledEdit("n:", "dimensión")
        dims2.addWidget(self.k2); dims2.addWidget(self.n2)
        g2.addLayout(dims2)
        self.V2 = MatrixInput("Lista de vectores (uno por línea, n valores)")
        self.c2 = MatrixInput("Coeficientes (k valores en una línea)", height=60)
        self.btn2 = btn("Calcular combinación")
        g2.addWidget(self.V2); g2.addWidget(self.c2); g2.addWidget(self.btn2)
        mlay.addWidget(group2)

        self.out = OutputArea(); mlay.addWidget(self.out)

        self.btn1.clicked.connect(self.on_ecuacion)
        self.btn2.clicked.connect(self.on_comb)

    def on_ecuacion(self):
        try:
            k = int(self.k1.text()); n = int(self.n1.text())
            V = parse_list_vectors(self.V1.text(), k, n)
            bvals = parse_nums(self.b1.text().replace("\n", " "))
            if len(bvals) != n: raise ValueError(f"b debe tener {n} valores.")
            out = ecuacion_vectorial(V, bvals)
            lines = []
            for p in out.get("reportes", []): lines.append(p + "\n")
            tipo = out.get("tipo") or out.get("estado")
            lines.append("Estado: " + str(tipo))
            lines.append("")
            lines.append(formatear_solucion_parametrica(out, nombres_vars=None, dec=4, fracciones=True))
            self.out.clear_and_write("\n".join(lines))
        except Exception as e:
            self.out.clear_and_write("Error: " + str(e))

    def on_comb(self):
        try:
            k = int(self.k2.text()); n = int(self.n2.text())
            V = parse_list_vectors(self.V2.text(), k, n)
            coef = parse_nums(self.c2.text().replace("\n", " "))
            if len(coef) != k: raise ValueError(f"Se esperaban {k} coeficientes.")
            out = combinacion_lineal_explicada(V, coef, dec=4)
            self.out.clear_and_write(out["texto"] + "\n\nComo lista: " + str(out["resultado"]))
        except Exception as e:
            self.out.clear_and_write("Error: " + str(e))

# =========================
#   Ventana principal
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Álgebra Lineal — GUI (PyQt6)")
        self.resize(1100, 780)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(TabGaussJordan(), "Gauss-Jordan (Ab)")
        tabs.addTab(TabAXeqB(), "AX=B")
        tabs.addTab(TabProg4(), "Programa 4")
        tabs.addTab(TabVectores(), "Vectores")

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Listo")

# =========================
#   main() con tema claro
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)

    # Tema claro (blanco-gris)
    palette = QtGui.QPalette()
    base_color = QtGui.QColor(245, 245, 245)
    text_color = QtGui.QColor(20, 20, 20)
    panel_color = QtGui.QColor(255, 255, 255)
    highlight = QtGui.QColor(41, 128, 185)

    palette.setColor(QtGui.QPalette.ColorRole.Window, base_color)
    palette.setColor(QtGui.QPalette.ColorRole.Base, panel_color)
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(235, 235, 235))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.Text, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(255, 255, 225))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, text_color)

    app.setPalette(palette)
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
