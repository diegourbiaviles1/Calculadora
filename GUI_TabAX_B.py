# GUI_TabAX_B.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from widgets import LabeledEdit, MatrixTable, VectorInputTable, OutputArea, btn, format_matrix_text
from algebra_vector import resolver_AX_igual_B

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
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Conexiones
        self.run.clicked.connect(self.on_run)
        self.btn_fmt.clicked.connect(self._toggle_fmt)
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

    def _set_output_text(self, base_text: str):
        self.out.set_output_format(base_text, self._show_frac)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        if hasattr(self.out, '_last_dec'):
            self.out.clear_and_write(self.out._last_frac if self._show_frac else self.out._last_dec)

    def on_run(self):
        try:
            A = self.A_table.to_matrix()
            b_vals = self.b_input.to_vector()
            
            # Asume que siempre es vector b (B de una columna) para simplificar la interfaz.
            out = resolver_AX_igual_B(A, b_vals)

            lines = []
            lines.append("--- Procedimiento de Gauss-Jordan ---")
            for p in out.get("reportes", []): lines.append(p)
            
            lines.append("\n--- Resultado ---")
            if out.get("estado") == "ok":
                if "x" in out: 
                    lines.append("Solución única encontrada:")
                    lines.append("x = " + str(out["x"]))
                if "X" in out and out["X"] is not None:
                    lines.append("Matriz solución X encontrada:")
                    lines.append(format_matrix_text(out["X"]))
            else:
                lines.append("Estado del sistema: " + str(out.get("estado")))
                lines.append("No se encontró una solución única.")
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error: {e}")