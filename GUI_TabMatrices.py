# GUI_TabMatrices.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from widgets import LabeledEdit, MatrixTable, OutputArea, btn, format_matrix_text, _safe_float
from matrices import suma_matrices_explicada, resta_matrices_explicada, producto_escalar_explicado, producto_matrices_explicado, traspuesta_explicada, propiedad_r_suma_traspuesta_explicada
from utilidad import evaluar_expresion, fmt_number, DEFAULT_DEC
from algebra_vector import columnas # Necesario para obtener la transpuesta para Prop. 5

# --- Helpers for element-wise explanation (extracted from app_gui.py) ---
def _num(x):  # uses the same format as the rest of the app
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

def explain_rsumT_left(A, B, r):
    m, n = len(A), len(A[0])
    lines = []
    for i in range(n):       # filas de L
        for j in range(m):   # cols de L
            aji = A[j][i]
            bji = B[j][i]
            suma = aji + bji
            val  = r * suma
            lines.append(
                f"L[{i+1},{j+1}] = r·(A[{j+1},{i+1}] + B[{j+1},{i+1}]) = "
                f"{_num(r)}·({_num(aji)} + {_num(bji)}) = {_num(r)}·{_num(suma)} = {_num(val)}"
            )
    return lines

def explain_rsumT_right(A, B, r):
    m, n = len(A), len(A[0])
    lines = []
    for i in range(n):       # filas de R
        for j in range(m):   # cols de R
            aT = A[j][i]
            bT = B[j][i]
            suma = aT + bT
            val  = r * suma
            lines.append(
                f"R[{i+1},{j+1}] = r·(A^T[{i+1},{j+1}] + B^T[{i+1},{j+1}]) = "
                f"{_num(r)}·(A[{j+1},{i+1}] + B[{j+1},{i+1}]) = "
                f"{_num(r)}·({_num(aT)} + {_num(bT)}) = {_num(r)}·{_num(suma)} = {_num(val)}"
            )
    return lines


class TabProg5(QtWidgets.QWidget):
    """Programa 5 — Operaciones con matrices y verificación de propiedades de la traspuesta."""
    def __init__(self, parent=None):
        super().__init__(parent)
        main = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.sp_m = QtWidgets.QSpinBox(); self.sp_m.setRange(1, 50); self.sp_m.setValue(2)
        top.addWidget(QtWidgets.QLabel("A: m=")); top.addWidget(self.sp_m)
        self.sp_n = QtWidgets.QSpinBox(); self.sp_n.setRange(1, 50); self.sp_n.setValue(2)
        top.addWidget(QtWidgets.QLabel(" n=")); top.addWidget(self.sp_n)
        top.addSpacing(16)
        self.sp_r = QtWidgets.QSpinBox(); self.sp_r.setRange(1, 50); self.sp_r.setValue(2)
        top.addWidget(QtWidgets.QLabel("B: m=")); top.addWidget(self.sp_r)
        self.sp_p = QtWidgets.QSpinBox(); self.sp_p.setRange(1, 50); self.sp_p.setValue(2)
        top.addWidget(QtWidgets.QLabel(" n=")); top.addWidget(self.sp_p)
        top.addSpacing(16)
        self.k_edit = QtWidgets.QLineEdit("1"); self.k_edit.setValidator(QtGui.QDoubleValidator())
        self.k_edit.setMaximumWidth(120)
        top.addWidget(QtWidgets.QLabel("r =")); top.addWidget(self.k_edit)
        top.addStretch(1)
        main.addLayout(top)
        
        # Disable spin wheel/buttons on creation (Assuming helper exists in widgets.py)
        # We need to manually call _disable_spin_wheel or use hide_all_spin_buttons later
        from widgets import _disable_spin_wheel
        _disable_spin_wheel(self.sp_m); _disable_spin_wheel(self.sp_n)
        _disable_spin_wheel(self.sp_r); _disable_spin_wheel(self.sp_p)
        self.sp_m.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.sp_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.sp_r.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.sp_p.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)


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
        self.btn_fmt = btn("Cambiar a fracciones")
        rl.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        center.addWidget(left)
        center.addWidget(right)
        center.setStretchFactor(0, 1)
        center.setStretchFactor(1, 1)
        main.addWidget(center)

        # Conexiones
        self.btn_fmt.clicked.connect(self._toggle_fmt)
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
    def _set_output_text(self, base_text: str):
        self.out.set_output_format(base_text, self._show_frac)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        if hasattr(self.out, '_last_dec'):
            self.out.clear_and_write(self.out._last_frac if self._show_frac else self.out._last_dec)

    def _sync_A(self):
        self.tblA.set_size(self.sp_m.value(), self.sp_n.value())

    def _sync_B(self):
        self.tblB.set_size(self.sp_r.value(), self.sp_p.value())

    def _fill_zeros(self):
        self.tblA.fill_zeros(); self.tblB.fill_zeros()

    def _clear_all(self):
        self.tblA.clear_all(); self.tblB.clear_all(); self._set_output_text("")

    def _k_value(self) -> float:
        t = self.k_edit.text().strip() or "0"
        return float(evaluar_expresion(t, exacto=False))

    def on_sum(self):
        try:
            A = self.tblA.to_matrix(); B = self.tblB.to_matrix()
            out = suma_matrices_explicada(A, B)
            detalle = explain_sum(A, B)

            lines = ["--- Pasos ---"] + out["pasos"]
            lines += ["", "Resultado C = A + B:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(A + B)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "A^T:", format_matrix_text(out["AT"]), "", "B^T:", format_matrix_text(out["BT"])]
            lines += ["", "A^T + B^T:", format_matrix_text(out["AT_mas_BT"]), "", ">>> " + out["conclusion"]]
            lines += ["", "--- Cálculo elemento a elemento (C[i,j]) ---"] + detalle
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error:\n{e}")

    def on_res(self):
        try:
            A = self.tblA.to_matrix(); B = self.tblB.to_matrix()
            out = resta_matrices_explicada(A, B)
            detalle = explain_res(A, B)

            lines = ["--- Pasos ---"] + out["pasos"]
            lines += ["", "Resultado C = A - B:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(A - B)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "A^T:", format_matrix_text(out["AT"]), "", "B^T:", format_matrix_text(out["BT"])]
            lines += ["", "A^T - B^T:", format_matrix_text(out["AT_menos_BT"]), "", ">>> " + out["conclusion"]]
            lines += ["", "--- Cálculo elemento a elemento (C[i,j]) ---"] + detalle
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error:\n{e}")

    def on_kA(self):
        try:
            A = self.tblA.to_matrix()
            r = self._k_value()
            out = producto_escalar_explicado(r, A)
            detalle = explain_kA(r, A)

            lines = ["--- Pasos ---"] + out["pasos"]
            lines += ["", "Resultado r·A:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(rA)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "A^T:", format_matrix_text(out["AT"]), "", "r·A^T:", format_matrix_text(out["kAT"])]
            lines += ["", ">>> " + out["conclusion"]]
            lines += ["", "--- Cálculo elemento a elemento (C[i,j]) ---"] + detalle
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error:\n{e}")

    def on_AB(self):
        try:
            if self.sp_r.value() != self.sp_n.value():
                raise ValueError("Para A·B se requiere filas(B) = columnas(A). Ajusta los spinners.")
            A = self.tblA.to_matrix(); B = self.tblB.to_matrix()
            out = producto_matrices_explicado(A, B)
            detalle = explain_AB(A, B)

            lines = ["--- Pasos ---"] + out["pasos"]
            lines += ["", "Resultado C = A·B:", format_matrix_text(out["resultado"]), ""]
            lines += ["--- Verificación de Propiedad ---", "(AB)^T:", format_matrix_text(out["traspuesta_del_resultado"])]
            lines += ["", "B^T:", format_matrix_text(out["BT"]), "", "A^T:", format_matrix_text(out["AT"])]
            lines += ["", "B^T·A^T:", format_matrix_text(out["BT_por_AT"]), "", ">>> " + out["conclusion"]]
            lines += ["", "--- Cálculo elemento a elemento (C[i,j]) ---"] + detalle
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error:\n{e}")

    # --- Resto de botones ---
    def on_AT(self):
        try:
            A = self.tblA.to_matrix()
            out = traspuesta_explicada(A)
            lines = ["--- Pasos ---", "Se intercambian filas por columnas."]
            lines += ["", "Resultado A^T:", format_matrix_text(out["resultado"])]
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error:\n{e}")

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
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error:\n{e}")

    def on_rSumT_prop(self):
        try:
            A = self.tblA.to_matrix(); B = self.tblB.to_matrix()
            r = self._k_value()
            out = propiedad_r_suma_traspuesta_explicada(A, B, r)

            det_izq = explain_rsumT_left(A, B, r)
            det_der = explain_rsumT_right(A, B, r)

            lines = ["--- Verificación de (r(A+B))^T = r(A^T+B^T) ---"]
            lines += out["pasos"]
            lines += [
                "", "Lado Izquierdo (r(A+B))^T:", format_matrix_text(out["izquierda"]),
                "", "Lado Derecho r(A^T+B^T):",  format_matrix_text(out["derecha"]),
                "", ">>> " + out["conclusion"],
                "", "--- Cálculo elemento a elemento (Lado Izquierdo) ---",
            ]
            lines += det_izq
            lines += ["", "--- Cálculo elemento a elemento (Lado Derecho) ---"]
            lines += det_der

            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error:\n{e}")