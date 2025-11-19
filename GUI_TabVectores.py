# GUI_TabVectores.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from widgets import LabeledEdit, MatrixTable, VectorInputTable, OutputArea, btn, format_matrix_text
from algebra_vector import verificar_propiedades, verificar_distributiva_matriz, combinacion_lineal_explicada, ecuacion_vectorial
from utilidad import evaluar_expresion

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
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Conexiones
        self.btn_fmt.clicked.connect(self._toggle_fmt)
        self.op_selector.currentIndexChanged.connect(self.stack.setCurrentIndex)
    
    # helpers de formato
    def _set_output_text(self, base_text: str):
        self.out.set_output_format(base_text, self._show_frac)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        if hasattr(self.out, '_last_dec'):
            self.out.clear_and_write(self.out._last_frac if self._show_frac else self.out._last_dec)
    
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
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error: {e}")

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
            self._set_output_text("\n".join(out["pasos"]))
        except Exception as e:
            self._set_output_text(f"Error: {e}")

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
            self._set_output_text(f"{out['texto']}\n\nComo lista: {out['resultado_simple']}")
        except Exception as e:
            self._set_output_text(f"Error: {e}")

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
            lines = [p for p in out.get("reportes", [])]
            lines.append(f"\nEstado: {out.get('tipo') or out.get('estado')}\n")
            # Note: formatear_solucion_parametrica handles its own formatting (fractions=True here)
            lines.append(out["salida_parametrica"])
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error: {e}")