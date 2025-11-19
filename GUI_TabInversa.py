# GUI_TabInversa.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from widgets import LabeledEdit, MatrixTable, OutputArea, btn, format_matrix_text
from inversa import inversa_por_gauss_jordan, verificar_propiedades_invertibilidad, programa_inversa_con_propiedades
from sistema_lineal import formatear_solucion_parametrica # Necesario para mostrar Ax=0 detalle
from utilidad import DEFAULT_EPS

class TabProg6(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)

        # --- fila superior ---
        top = QtWidgets.QHBoxLayout()
        self.sp_n = QtWidgets.QSpinBox(); self.sp_n.setRange(1, 50); self.sp_n.setValue(2)
        top.addWidget(QtWidgets.QLabel("n:")); top.addWidget(self.sp_n)
        top.addStretch(1)
        root.addLayout(top)
        
        # Disable spin wheel/buttons on creation (Assuming helper exists in widgets.py)
        from widgets import _disable_spin_wheel
        _disable_spin_wheel(self.sp_n)
        self.sp_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)


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
        self.btn_fmt = btn("Cambiar a fracciones")
        root.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # conexiones / atajos
        self.sp_n.valueChanged.connect(self._sync_n)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_inv.clicked.connect(self.on_inv)
        self.btn_props.clicked.connect(self.on_props)
        self.btn_all.clicked.connect(self.on_all)
        self.btn_fmt.clicked.connect(self._toggle_fmt)

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
        self.summary.set_output_format(base_text, self._show_frac)

    def _set_steps(self, base_text: str):
        self.steps_all.set_output_format(base_text, self._show_frac)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        if hasattr(self.summary, '_last_dec'):
            self.summary.clear_and_write(self.summary._last_frac if self._show_frac else self.summary._last_dec)
        if hasattr(self.steps_all, '_last_dec'):
            self.steps_all.clear_and_write(self.steps_all._last_frac if self._show_frac else self.steps_all._last_dec)


    # ---------- acciones ----------
    def on_inv(self):
        try:
            A = self._A()
            inv = inversa_por_gauss_jordan(A, tol=DEFAULT_EPS, dec=4)

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
            self._set_summary(f"Error: {e}")
            self._set_steps("")

    def on_props(self):
        try:
            A = self._A()
            props = verificar_propiedades_invertibilidad(A, tol=DEFAULT_EPS, dec=4)

            lines = []
            lines += ["=== Propiedades (c)(d)(e) ===", ""]
            lines += [f"Pivotes (1-index): {props.get('pivotes', 'N/A')}",
                      f"Rango: {props.get('rango', 'N/A')}", ""]
            lines += [props.get("explicacion", "No se pudo generar la explicación.")]
            
            if props.get("detalle_sistema_homogeneo"):
                lines.append("\n--- Detalles del sistema homogéneo Ax=0 ---")
                param_txt = formatear_solucion_parametrica(props["detalle_sistema_homogeneo"], dec=4, fracciones=True)
                lines.append(param_txt)
                
            self._set_summary("\n".join(lines))
            self._set_steps("(Ejecuta 'Calcular A^{-1} y |A|' o 'Todo junto' para ver el procedimiento).")

        except Exception as e:
            self._set_summary(f"Error: {e}")
            self._set_steps("")

    def on_all(self):
        try:
            A = self._A()
            full = programa_inversa_con_propiedades(A, tol=DEFAULT_EPS, dec=4)
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
            self._set_summary(f"Error: {e}")
            self._set_steps("")