# GUI_TabDeterminante.py
from __future__ import annotations
from ast import List
from PyQt6 import QtWidgets, QtCore, QtGui
from widgets import LabeledEdit, MatrixTable, VectorInputTable, OutputArea, btn
from determinante import det_cofactores, det_sarrus, det_por_cramer, interpretar_invertibilidad, \
                         propiedad_fila_col_cero, propiedad_filas_prop, propiedad_swap_signo, \
                         propiedad_multiplicar_fila, propiedad_multiplicativa
from utilidad import evaluar_expresion, DEFAULT_DEC
from algebra_vector import columnas # Necesario para obtener la transpuesta para Prop. 5

class TabDeterminante(QtWidgets.QWidget):
    """Programa 7 — Determinante: Cofactores, Sarrus, Cramer + Propiedades."""
    def __init__(self, parent=None):
        super().__init__(parent)
        main = QtWidgets.QVBoxLayout(self)

        # --- Dimensión ---
        dims = QtWidgets.QHBoxLayout()
        self.sp_n = QtWidgets.QSpinBox(); self.sp_n.setRange(1, 8); self.sp_n.setValue(3)
        dims.addWidget(QtWidgets.QLabel("n (cuadrada):"))
        dims.addWidget(self.sp_n)
        dims.addStretch(1)
        main.addLayout(dims)

        # --- Selector de método ---
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["Cofactores (general)", "Sarrus (3×3)", "Cramer (ilustrativo)"])
        main.addWidget(self.method)

        # --- k para Propiedad 4 (editable) ---
        row_k = QtWidgets.QHBoxLayout()
        self.k_prop4 = LabeledEdit("k (Prop. 4):", "3", default_value="3")
        row_k.addWidget(self.k_prop4)
        row_k.addStretch(1)
        main.addLayout(row_k)

        # --- Vector b (opcional para Cramer) ---
        self.b_input = VectorInputTable("Vector b (opcional para Cramer)", 3)
        self.b_input.table.setFixedHeight(72)
        hh = self.b_input.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setMinimumSectionSize(90)
        main.addWidget(self.b_input)

        # --- Matriz A ---
        self.tblA = MatrixTable(3, 3, "Matriz A (n×n)")
        main.addWidget(self.tblA)

        # --- Botones ---
        btns = QtWidgets.QHBoxLayout()
        self.btn_det = btn("Calcular determinante")
        btns.addWidget(self.btn_det)

        # Selector de Propiedad
        self.prop_selector = QtWidgets.QComboBox()
        self.prop_selector.addItems([
            "Prop. 1: Fila/Columna Cero",
            "Prop. 2: Filas Proporcionales",
            "Prop. 3: Intercambio de Filas (Signo)",
            "Prop. 4: Multiplicar Fila por k",
            "Prop. 5: det(AB) = det(A)det(B) (con B=A^T)",
        ])
        self.btn_run_prop = btn("Verificar Propiedad")
        
        btns.addSpacing(20) # Espacio separador
        btns.addWidget(self.prop_selector)
        btns.addWidget(self.btn_run_prop)
        btns.addStretch(1)
        main.addLayout(btns)

        # --- Salida ---
        self.out = OutputArea()
        main.addWidget(self.out)

        # --- Botón fracciones/decimales (controlado por la clase) ---
        self._show_frac = False
        self.btn_fmt = btn("Cambiar a fracciones")
        main.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Eventos / sincronización
        self.sp_n.valueChanged.connect(self._sync_n)
        self.btn_det.clicked.connect(self.on_det)
        self.btn_run_prop.clicked.connect(self.on_props)
        self.btn_fmt.clicked.connect(self._toggle_fmt)

        # Inicial
        self._sync_n()

    def _sync_n(self):
        n = int(self.sp_n.value())
        self.tblA.set_size(n, n)
        self.b_input.set_size(n)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        if hasattr(self.out, '_last_dec'):
             self.out.clear_and_write(self.out._last_frac if self._show_frac else self.out._last_dec)
        
    def _set_output_text(self, lines: List[str]):
        base_text = "\n".join(lines)
        if hasattr(self.out, 'set_output_format'):
             self.out.set_output_format(base_text, self._show_frac)
        else:
             self.out.clear_and_write(base_text)


    def on_det(self):
        try:
            A = self.tblA.to_matrix()
            metodo = self.method.currentText()
            lines = []

            if metodo.startswith("Cofactores"):
                out = det_cofactores(A, dec=4)
                detA = out["det"]; lines += ["[Cofactores]\n"] + out["pasos"]
            elif metodo.startswith("Sarrus"):
                out = det_sarrus(A, dec=4)
                detA = out["det"]; lines += ["[Sarrus]\n"] + out["pasos"]
            else:  # Cramer
                b = self.b_input.to_vector()
                out = det_por_cramer(A, b, dec=4)
                detA = out["det"]; lines += ["[Cramer]\n"] + out["pasos"]

            lines += ["", f"Conclusión: {out['conclusion']}",
                      interpretar_invertibilidad(detA, dec=4)]
            self._set_output_text(lines)

        except Exception as e:
            self.out.clear_and_write(f"Error: {e}")

    def on_props(self):
        try:
            A = self.tblA.to_matrix()
            
            selected_prop = self.prop_selector.currentText()
            lines = []
            
            if "Prop. 1" in selected_prop:
                p1 = propiedad_fila_col_cero(A, dec=4)
                lines += ["[Propiedad 1] Fila/columna cero → det(A)=0."] + p1["pasos"] + [p1["conclusion"]]
            
            elif "Prop. 2" in selected_prop:
                p2 = propiedad_filas_prop(A, dec=4)
                lines += ["[Propiedad 2] Filas/columnas proporcionales → det(A)=0."] + p2["pasos"] + [p2["conclusion"]]
            
            elif "Prop. 3" in selected_prop:
                p3 = propiedad_swap_signo(A, dec=4)
                lines += ["[Propiedad 3] Intercambio de filas cambia el signo."] + p3["pasos"] + [p3["conclusion"]]
            
            elif "Prop. 4" in selected_prop:
                try:
                    k_val = float(evaluar_expresion(self.k_prop4.text()))
                except Exception as e:
                    raise ValueError(f"Escalar k inválido: {e}")
                p4 = propiedad_multiplicar_fila(A, k_val, dec=4)
                lines += [f"[Propiedad 4] Multiplicar fila por k escala det (k={k_val})."] + p4["pasos"] + [p4["conclusion"]]
            
            elif "Prop. 5" in selected_prop:
                # B por defecto = A^T
                # Nota: usamos columnas(A) y copiamos para transponer
                B = [fila[:] for fila in columnas(A)] 
                p5 = propiedad_multiplicativa(A, B, dec=4)
                lines += ["[Propiedad 5] det(AB) = det(A)·det(B) (con B=A^T)."] + p5["pasos"] + [p5["conclusion"]]
            
            else:
                lines.append("Por favor, seleccione una propiedad.")

            self._set_output_text(lines)

        except Exception as e:
            self.out.clear_and_write(f"Error: {e}")