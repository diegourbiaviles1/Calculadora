# GUI_TabMetodosNumericos.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from widgets import LabeledEdit, OutputArea, btn
from metodos_numericos import biseccion_descriptiva, regla_falsa_descriptiva, calcular_errores, tipos_de_error_texto, ejemplo_punto_flotante_texto, propagacion_error
from notacion_posicional import descomponer_base10, descomponer_base2
from utilidad import fmt_number, DEFAULT_DEC

class TabMetodosNumericos(QtWidgets.QWidget):
    """Pestaña: Métodos numéricos y errores."""
    def __init__(self, parent=None):
        super().__init__(parent)
        main = QtWidgets.QVBoxLayout(self)

        # ====== Bloque 1: Raíces de ecuaciones no lineales ======
        grp_raices = QtWidgets.QGroupBox("Raíces de ecuaciones no lineales — Métodos numéricos")
        v_raices = QtWidgets.QVBoxLayout(grp_raices)

        fila_fx = QtWidgets.QHBoxLayout()
        self.ed_expr = QtWidgets.QLineEdit("x^3 - x - 1")
        fila_fx.addWidget(QtWidgets.QLabel("f(x) ="))
        fila_fx.addWidget(self.ed_expr)
        v_raices.addLayout(fila_fx)

        fila_int = QtWidgets.QHBoxLayout()
        self.ed_a = QtWidgets.QLineEdit("1")
        self.ed_b = QtWidgets.QLineEdit("2")
        self.ed_tol = QtWidgets.QLineEdit("1e-4")
        self.ed_max = QtWidgets.QLineEdit("50")

        fila_int.addWidget(QtWidgets.QLabel("Extremo izquierdo a:"))
        fila_int.addWidget(self.ed_a)
        fila_int.addSpacing(10)
        fila_int.addWidget(QtWidgets.QLabel("Extremo derecho b:"))
        fila_int.addWidget(self.ed_b)
        fila_int.addSpacing(10)
        fila_int.addWidget(QtWidgets.QLabel("Tolerancia:"))
        fila_int.addWidget(self.ed_tol)
        fila_int.addSpacing(10)
        fila_int.addWidget(QtWidgets.QLabel("Máx. iteraciones:"))
        fila_int.addWidget(self.ed_max)
        v_raices.addLayout(fila_int)

        fila_met = QtWidgets.QHBoxLayout()
        fila_met.addWidget(QtWidgets.QLabel("Método:"))
        self.cbo_metodo = QtWidgets.QComboBox()
        self.cbo_metodo.addItems(["Bisección", "Regla falsa"])
        fila_met.addWidget(self.cbo_metodo)
        fila_met.addStretch(1)
        v_raices.addLayout(fila_met)

        self.btn_calcular_raiz = btn("Calcular raíz y tabla de errores")
        self.btn_calcular_raiz.clicked.connect(self.on_calcular_raiz)
        v_raices.addWidget(self.btn_calcular_raiz)

        self.out_raices = OutputArea()
        v_raices.addWidget(self.out_raices)

        main.addWidget(grp_raices)

        # ====== Bloque 2: Errores numéricos ======
        grp_err = QtWidgets.QGroupBox("Errores numéricos | Error absoluto, relativo y tipos de error")
        v_err = QtWidgets.QVBoxLayout(grp_err)

        fila_val = QtWidgets.QHBoxLayout()
        self.ed_val_real = QtWidgets.QLineEdit("3.1416")
        self.ed_val_aprox = QtWidgets.QLineEdit("3.14")
        fila_val.addWidget(QtWidgets.QLabel("Valor real m:"))
        fila_val.addWidget(self.ed_val_real)
        fila_val.addSpacing(10)
        fila_val.addWidget(QtWidgets.QLabel("Valor aproximado x:"))
        fila_val.addWidget(self.ed_val_aprox)
        v_err.addLayout(fila_val)

        fila_btn_err = QtWidgets.QHBoxLayout()
        self.btn_calc_err = btn("Calcular ea, er y er%")
        self.btn_calc_err.clicked.connect(self.on_calcular_errores)
        self.btn_tipos_err = btn("Mostrar tipos de error", kind="ghost")
        self.btn_tipos_err.clicked.connect(self.on_tipos_error)
        self.btn_ej_flot = btn("Ejemplo de punto flotante", kind="ghost")
        self.btn_ej_flot.clicked.connect(self.on_ejemplo_flotante)
        fila_btn_err.addWidget(self.btn_calc_err)
        fila_btn_err.addWidget(self.btn_tipos_err)
        fila_btn_err.addWidget(self.btn_ej_flot)
        fila_btn_err.addStretch(1)
        v_err.addLayout(fila_btn_err)

        self.out_err = OutputArea()
        v_err.addWidget(self.out_err)

        main.addWidget(grp_err)

        # ====== Bloque 3: Notación posicional ======
        grp_not = QtWidgets.QGroupBox("Notación posicional (base 10 y base 2)")
        v_not = QtWidgets.QVBoxLayout(grp_not)

        fila_n = QtWidgets.QHBoxLayout()
        self.ed_entero = QtWidgets.QLineEdit("472")
        fila_n.addWidget(QtWidgets.QLabel("Número entero:"))
        fila_n.addWidget(self.ed_entero)
        v_not.addLayout(fila_n)

        fila_btn_not = QtWidgets.QHBoxLayout()
        self.btn_base10 = btn("Descomponer en base 10")
        self.btn_base10.clicked.connect(self.on_base10)
        self.btn_base2 = btn("Descomponer en base 2")
        self.btn_base2.clicked.connect(self.on_base2)
        fila_btn_not.addWidget(self.btn_base10)
        fila_btn_not.addWidget(self.btn_base2)
        fila_btn_not.addStretch(1)
        v_not.addLayout(fila_btn_not)

        self.out_not = OutputArea()
        v_not.addWidget(self.out_not)

        main.addWidget(grp_not)

        # ====== Bloque 4: Propagación del error ======
        grp_prop = QtWidgets.QGroupBox("Propagación del error en una función f(x)")
        v_prop = QtWidgets.QVBoxLayout(grp_prop)

        fila_fx_prop = QtWidgets.QHBoxLayout()
        self.ed_fx_prop = QtWidgets.QLineEdit("x^2 + 3x")
        fila_fx_prop.addWidget(QtWidgets.QLabel("f(x) para propagación:"))
        fila_fx_prop.addWidget(self.ed_fx_prop)
        v_prop.addLayout(fila_fx_prop)

        fila_x0 = QtWidgets.QHBoxLayout()
        self.ed_x0 = QtWidgets.QLineEdit("2")
        self.ed_dx = QtWidgets.QLineEdit("0.01")
        fila_x0.addWidget(QtWidgets.QLabel("Valor central x0:"))
        fila_x0.addWidget(self.ed_x0)
        fila_x0.addSpacing(10)
        fila_x0.addWidget(QtWidgets.QLabel("Error en x (Δx):"))
        fila_x0.addWidget(self.ed_dx)
        v_prop.addLayout(fila_x0)

        self.btn_prop = btn("Calcular propagación del error")
        self.btn_prop.clicked.connect(self.on_propagacion)
        v_prop.addWidget(self.btn_prop)

        self.out_prop = OutputArea()
        v_prop.addWidget(self.out_prop)

        main.addWidget(grp_prop)
        main.addStretch(1)

    # ================== Handlers ==================
    def _tabla_pasos(self, info):
        # Helper interno para formatear la tabla de pasos de Bisección/Regla Falsa
        pasos = info.get("pasos", [])
        if not pasos:
            return "No se generaron iteraciones."

        header = (
            "Iter   a          b          xr         f(xr)        ea           er           er%\n"
            "-------------------------------------------------------------------------------"
        )
        filas = [header]
        for p in pasos:
            it = p["numero_de_iteracion"]
            a = p["extremo_izquierdo_a"]
            b = p["extremo_derecho_b"]
            xr = p["aproximacion_actual_xr"]
            fxr = p["valor_de_la_funcion_en_xr"]
            ea = p["error_absoluto_ea"]
            er = p["error_relativo"]
            erp = p["error_relativo_porcentual"]

            def fval(v):
                if v is None:
                    return "   ---   "
                return fmt_number(float(v), DEFAULT_DEC, False).rjust(10)

            fila = f"{it:>3}  {fval(a)} {fval(b)} {fval(xr)} {fval(fxr)} {fval(ea)} {fval(er)} {fval(erp)}"
            filas.append(fila)

        return "\n".join(filas)

    def on_calcular_raiz(self):
        try:
            expr = self.ed_expr.text().strip()
            a = float(self.ed_a.text())
            b = float(self.ed_b.text())
            tol = float(self.ed_tol.text())
            max_iter = int(self.ed_max.text())

            if self.cbo_metodo.currentText().startswith("Bisección"):
                info = biseccion_descriptiva(expr, a, b, tol, max_iter)
            else:
                info = regla_falsa_descriptiva(expr, a, b, tol, max_iter)

            lineas = []
            lineas.append(f"Método: {info['metodo']}")
            lineas.append(f"f(x) = {expr}")
            lineas.append(f"Intervalo inicial: [{a}, {b}]")
            lineas.append(f"Tolerancia: {tol}")
            lineas.append("")
            lineas.append(self._tabla_pasos(info))
            lineas.append("")
            lineas.append(
                "Raíz aproximada: "
                + fmt_number(info["raiz_aproximada"], DEFAULT_DEC, False)
            )
            lineas.append(
                "f(raíz) ≈ "
                + fmt_number(info["valor_funcion_en_raiz"], DEFAULT_DEC, False)
            )

            self.out_raices.clear_and_write("\n".join(lineas))
        except Exception as e:
            self.out_raices.clear_and_write(f"Error: {e}")

    def on_calcular_errores(self):
        try:
            m = float(self.ed_val_real.text())
            x = float(self.ed_val_aprox.text())
            info = calcular_errores(m, x)
            ea = fmt_number(info["error_absoluto"], DEFAULT_DEC, False)
            er = "---" if info["error_relativo"] is None else fmt_number(info["error_relativo"], DEFAULT_DEC, False)
            erp = "---" if info["error_relativo_porcentual"] is None else fmt_number(info["error_relativo_porcentual"], DEFAULT_DEC, False)

            texto = (
                f"Valor real m = {info['valor_real']}\n"
                f"Valor aproximado x = {info['valor_aproximado']}\n\n"
                f"Error absoluto ea = |m - x| = {ea}\n"
                f"Error relativo er = ea / |m| = {er}\n"
                f"Error relativo porcentual er% = er × 100 = {erp}"
            )
            self.out_err.clear_and_write(texto)
        except Exception as e:
            self.out_err.clear_and_write(f"Error: {e}")

    def on_tipos_error(self):
        self.out_err.clear_and_write(tipos_de_error_texto())

    def on_ejemplo_flotante(self):
        self.out_err.clear_and_write(ejemplo_punto_flotante_texto())

    def on_base10(self):
        try:
            n = int(self.ed_entero.text())
            self.out_not.clear_and_write(descomponer_base10(n))
        except Exception as e:
            self.out_not.clear_and_write(f"Error: {e}")

    def on_base2(self):
        try:
            n = int(self.ed_entero.text())
            self.out_not.clear_and_write(descomponer_base2(n))
        except Exception as e:
            self.out_not.clear_and_write(f"Error: {e}")

    def on_propagacion(self):
        try:
            expr = self.ed_fx_prop.text().strip()
            x0 = float(self.ed_x0.text())
            dx = float(self.ed_dx.text())
            info = propagacion_error(expr, x0, dx)
            texto = (
                f"f(x) = {expr}\n"
                f"x0 = {info['x0']}, Δx = {info['delta_x']}\n\n"
                f"f(x0) = {fmt_number(info['y0'], DEFAULT_DEC, False)}\n"
                f"f(x0 + Δx) = {fmt_number(info['y1'], DEFAULT_DEC, False)}\n"
                f"Δy ≈ f(x0 + Δx) − f(x0) = {fmt_number(info['delta_y'], DEFAULT_DEC, False)}\n\n"
                f"Error absoluto en y: {fmt_number(info['error_absoluto_y'], DEFAULT_DEC, False)}\n"
                f"Error relativo en y: "
                f"{'---' if info['error_relativo_y'] is None else fmt_number(info['error_relativo_y'], DEFAULT_DEC, False)}\n"
                f"Error relativo porcentual en y: "
                f"{'---' if info['error_relativo_porcentual_y'] is None else fmt_number(info['error_relativo_porcentual_y'], DEFAULT_DEC, False)}"
            )
            self.out_prop.clear_and_write(texto)
        except Exception as e:
            self.out_prop.clear_and_write(f"Error: {e}")