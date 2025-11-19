# GUI_TabDependencia.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from widgets import LabeledEdit, MatrixTable, VectorInputTable, OutputArea, btn
from homogeneo import resolver_sistema_homogeneo_y_no_homogeneo, resolver_dependencia_lineal_con_homogeneo, analizar_dependencia

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
        self.btn_fmt = btn("Cambiar a fracciones")
        mlay.addWidget(self.btn_fmt, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Conexiones
        self.btn_fmt.clicked.connect(self._toggle_fmt)
        self.op_selector.currentIndexChanged.connect(self.stack.setCurrentIndex)

    def _set_output_text(self, base_text: str):
        self.out.set_output_format(base_text, self._show_frac)

    def _toggle_fmt(self):
        self._show_frac = not self._show_frac
        self.btn_fmt.setText("Cambiar a decimales" if self._show_frac else "Cambiar a fracciones")
        if hasattr(self.out, '_last_dec'):
            self.out.clear_and_write(self.out._last_frac if self._show_frac else self.out._last_dec)

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
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error: {e}")

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

    def on_run_dependencia(self):
        try:
            V = self.V_table_dep.to_matrix()
            info = analizar_dependencia(V)
            lines = ["=== Conclusión ===", info["mensaje"]]
            lines += ["\n=== Pasos (Gauss-Jordan sobre A·c = 0, donde las columnas de A son los vectores vᵢ) ===\n"] + [p for p in info["pasos"]]
            lines += ["\n=== Coeficientes 'c' para la combinación lineal nula ===", info["salida_parametrica"]]
            self._set_output_text("\n".join(lines))
        except Exception as e:
            self._set_output_text(f"Error: {e}")