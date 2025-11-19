# GUI_TabGaussJordan.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from sistema_lineal import SistemaLineal, formatear_solucion_parametrica
from widgets import MatrixAugTable, OutputArea, btn, text_to_decimals, text_to_fractions

# --- Función Helper para la vista previa (extraída de app_gui.py) ---
def pretty_augmented_from_table(table: "MatrixAugTable", dec: int = 4) -> str:
    # Esta función se extrajo de su scope original y asume el acceso a la estructura de MatrixAugTable
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
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Conexiones
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

    def _set_output_text(self, base_text: str):
        self.out.set_output_format(base_text, self._show_frac)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        if hasattr(self.out, '_last_dec'):
            self.out.clear_and_write(self.out._last_frac if self._show_frac else self.out._last_dec)

    def on_run_table(self):
        try:
            Ab = self.aug_table.to_augmented()
            sl = SistemaLineal(Ab, decimales=4)
            out = sl.gauss_jordan()

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
            
            self._set_output_text("\n".join(lines + [param_txt]))
        except Exception as e:
            self._set_output_text(f"Error: {e}")