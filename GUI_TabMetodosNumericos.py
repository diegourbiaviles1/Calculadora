
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui

import math
import sympy as sp

from widgets import LabeledEdit, OutputArea, btn
from metodos_numericos import (
    biseccion_descriptiva,
    regla_falsa_descriptiva,
    calcular_errores,
    tipos_de_error_texto,
    ejemplo_punto_flotante_texto,
    propagacion_error,
    generar_reporte_paso_a_paso  # Para métodos cerrados
)
from notacion_posicional import descomponer_base10, descomponer_base2
from utilidad import fmt_number, DEFAULT_DEC

# Símbolo global para Sympy
x = sp.symbols("x")


# ==================================================
# Funciones internas para Newton y Secante (MODIFICADAS)
# ==================================================
def _parse_funcion(expr_str: str):
    """
    Convierte un string como 'x^3 - x - 1' en una función numérica f(x)
    y la expresión simbólica de Sympy.
    """
    expr_str = expr_str.replace("^", "**")
    # Manejo básico de e^x para sympy
    expr_str = expr_str.replace("e^", "exp") 
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, modules=["math"])
    return f, expr


def newton_raphson_descriptiva(expr_str: str, x0: float, tol: float, max_iter: int):
    """
    Método de Newton–Raphson con datos extra para el reporte detallado.
    """
    f, expr = _parse_funcion(expr_str)
    df_expr = sp.diff(expr, x)
    df = sp.lambdify(x, df_expr, modules=["math"])

    pasos = []
    xr_anterior = None
    xr = float(x0)
    criterio = "max_iter"
    ea = None

    # Guardamos la derivada como string para el reporte
    df_str = str(df_expr).replace("**", "^")

    for it in range(1, max_iter + 1):
        fx = f(xr)
        dfx = df(xr)

        if dfx == 0:
            criterio = "derivada_cero"
            break

        xr_nuevo = xr - fx / dfx

        if xr_anterior is None:
            ea = None
            er = None
            erp = None
        else:
            ea = abs(xr_nuevo - xr)
            er = ea / abs(xr_nuevo) if xr_nuevo != 0 else None
            erp = er * 100 if er is not None else None

        pasos.append({
            "numero_de_iteracion": it,
            "xi": xr,                 # Valor actual
            "f_xi": fx,               # f(xi)
            "df_xi": dfx,             # f'(xi)
            "xi_nuevo": xr_nuevo,     # El resultado de la fórmula
            "error_absoluto_ea": ea,
            "error_relativo": er,
            "error_relativo_porcentual": erp,
        })

        if ea is not None and ea <= tol:
            criterio = "tolerancia"
            xr = xr_nuevo
            break

        xr_anterior = xr
        xr = xr_nuevo

    raiz = xr
    valor_en_raiz = f(raiz)

    info = {
        "metodo": "Newton-Raphson",
        "expresion": expr_str,
        "derivada_str": df_str,
        "raiz_aproximada": raiz,
        "valor_funcion_en_raiz": valor_en_raiz,
        "iteraciones_totales": len(pasos),
        "criterio_de_paro": criterio,
        "pasos": pasos,
    }
    return info


def secante_descriptiva(expr_str: str, x0: float, x1: float, tol: float, max_iter: int):
    """
    Método de la Secante con datos extra para el reporte detallado.
    """
    f, _ = _parse_funcion(expr_str)

    pasos = []
    xr_ant = float(x0) # x_{i-1}
    xr = float(x1)     # x_i
    criterio = "max_iter"

    for it in range(1, max_iter + 1):
        f_xr = f(xr)
        f_xr_ant = f(xr_ant)

        denominador = (f_xr - f_xr_ant)
        if denominador == 0:
            criterio = "denominador_cero"
            break 

        xr_nuevo = xr - f_xr * (xr - xr_ant) / denominador

        ea = abs(xr_nuevo - xr)
        er = ea / abs(xr_nuevo) if xr_nuevo != 0 else None
        erp = er * 100 if er is not None else None

        pasos.append({
            "numero_de_iteracion": it,
            "xi_ant": xr_ant,         # x_{i-1}
            "xi": xr,                 # x_i
            "f_xi_ant": f_xr_ant,     # f(x_{i-1})
            "f_xi": f_xr,             # f(x_i)
            "xi_nuevo": xr_nuevo,     # Resultado
            "error_absoluto_ea": ea,
            "error_relativo": er,
            "error_relativo_porcentual": erp,
        })

        if ea <= tol:
            criterio = "tolerancia"
            xr = xr_nuevo
            break

        xr_ant = xr
        xr = xr_nuevo

    raiz = xr
    valor_en_raiz = f(raiz)

    info = {
        "metodo": "Secante",
        "expresion": expr_str,
        "raiz_aproximada": raiz,
        "valor_funcion_en_raiz": valor_en_raiz,
        "iteraciones_totales": len(pasos),
        "criterio_de_paro": criterio,
        "pasos": pasos,
    }
    return info

# ==================================================
# Generador de reporte para métodos abiertos
# ==================================================
def generar_reporte_abierto(info: dict) -> str:
    """Genera el texto paso a paso para Newton y Secante."""
    metodo = info["metodo"]
    expr_visual = info["expresion"].replace("**", "^")
    pasos = info["pasos"]
    
    lines = []
    lines.append(f"=== REPORTE DETALLADO: {metodo.upper()} ===")
    lines.append(f"Función: f(x) = {expr_visual}")
    
    if metodo == "Newton-Raphson":
        lines.append(f"Derivada analítica: f'(x) = {info['derivada_str']}")
        lines.append("-" * 60)
        
        for p in pasos:
            i = p["numero_de_iteracion"]
            xi = p["xi"]
            f_xi = p["f_xi"]
            df_xi = p["df_xi"]
            res = p["xi_nuevo"]
            ea = p["error_absoluto_ea"]
            
            s_xi = fmt_number(xi, 6)
            s_f = fmt_number(f_xi, 6)
            s_df = fmt_number(df_xi, 6)
            s_res = fmt_number(res, 6)
            s_ea = fmt_number(ea, 7) if ea is not None else "---"
            
            lines.append(f"\nIteración {i} (desde x{i-1} = {s_xi}):")
            lines.append(f"   1. Evaluaciones:")
            lines.append(f"      f({s_xi}) = {s_f}")
            lines.append(f"      f'({s_xi}) = {s_df}")
            lines.append(f"   2. Sustituir en fórmula de Newton:")
            lines.append(f"      x{i} = {s_xi} - ({s_f}) / ({s_df})")
            lines.append(f"         ≈ {s_res}")
            
            if ea is not None:
                lines.append(f"   3. Error aproximado:")
                lines.append(f"      Ea = |{s_res} - {s_xi}| ≈ {s_ea}")
            lines.append("-" * 40)

    elif metodo == "Secante":
        lines.append("-" * 60)
        for p in pasos:
            i = p["numero_de_iteracion"]
            xi_ant = p["xi_ant"]
            xi = p["xi"]
            fi_ant = p["f_xi_ant"]
            fi = p["f_xi"]
            res = p["xi_nuevo"]
            ea = p["error_absoluto_ea"]
            
            s_xia = fmt_number(xi_ant, 6)
            s_xi  = fmt_number(xi, 6)
            s_fa  = fmt_number(fi_ant, 6)
            s_fi  = fmt_number(fi, 6)
            s_res = fmt_number(res, 6)
            s_ea  = fmt_number(ea, 7) if ea is not None else "---"
            
            lines.append(f"\nIteración {i} (usar x{i-2}={s_xia}, x{i-1}={s_xi}):")
            lines.append(f"   1. Evaluaciones:")
            lines.append(f"      f({s_xia}) ≈ {s_fa}")
            lines.append(f"      f({s_xi})  ≈ {s_fi}")
            lines.append(f"   2. Sustituir en la fórmula de la secante:")
            # Formula visual: xi - f(xi)*(xi - x_ant) / (f(xi) - f(x_ant))
            num = f"({s_fi}) * ({s_xi} - {s_xia})"
            den = f"({s_fi} - {s_fa})"
            lines.append(f"      x{i} = {s_xi} - [ {num} / {den} ]")
            lines.append(f"         ≈ {s_res}")
            
            if ea is not None:
                lines.append(f"   Error aproximado respecto al paso anterior:")
                lines.append(f"      Ea = |{s_res} - {s_xi}| ≈ {s_ea}")
            lines.append("-" * 40)
            
    return "\n".join(lines)


# ==================================================
# Clase principal de la pestaña
# ==================================================
class TabMetodosNumericos(QtWidgets.QWidget):
    """
    Pestaña: Métodos numéricos, errores y notación posicional.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Caches para guardar info y poder generar reporte detallado
        self.info_raices_cache = None
        self.info_newton_cache = None
        self.info_secante_cache = None

        main = QtWidgets.QVBoxLayout(self)

        # Contenedor de subpestañas
        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs)

        # --------------------------------------------------
        # Tab 1: Métodos de raíces (cerrados)
        # --------------------------------------------------
        tab_raices = QtWidgets.QWidget()
        lay_raices = QtWidgets.QVBoxLayout(tab_raices)

        grp_raices = QtWidgets.QGroupBox("Raíces de ecuaciones no lineales — Métodos cerrados (Bisección / Regla falsa)")
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
        fila_int.addWidget(QtWidgets.QLabel("Error deseado (tolerancia):"))
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

        # Botones Tab 1
        self.btn_calcular_raiz = btn("Calcular raíz y tabla de errores")
        self.btn_calcular_raiz.clicked.connect(self.on_calcular_raiz)
        v_raices.addWidget(self.btn_calcular_raiz)

        self.btn_paso_a_paso = btn("Ver Paso a Paso (Detallado)", kind="ghost")
        self.btn_paso_a_paso.clicked.connect(self.on_mostrar_detalles)
        self.btn_paso_a_paso.setEnabled(False)
        v_raices.addWidget(self.btn_paso_a_paso)

        self.out_raices = OutputArea()
        v_raices.addWidget(self.out_raices)

        lay_raices.addWidget(grp_raices)
        lay_raices.addStretch(1)

        self.tabs.addTab(tab_raices, "Métodos cerrados")

        # --------------------------------------------------
        # Tab 2: Newton-Raphson (método abierto)
        # --------------------------------------------------
        tab_newton = QtWidgets.QWidget()
        lay_newton = QtWidgets.QVBoxLayout(tab_newton)

        grp_newton = QtWidgets.QGroupBox("Método abierto: Newton-Raphson")
        v_newton = QtWidgets.QVBoxLayout(grp_newton)

        fila_fx_n = QtWidgets.QHBoxLayout()
        self.ed_expr_newton = QtWidgets.QLineEdit("x^3 - x - 1")
        fila_fx_n.addWidget(QtWidgets.QLabel("f(x) ="))
        fila_fx_n.addWidget(self.ed_expr_newton)
        v_newton.addLayout(fila_fx_n)

        fila_newton = QtWidgets.QHBoxLayout()
        self.ed_x0_newton = QtWidgets.QLineEdit("1.5")
        self.ed_tol_newton = QtWidgets.QLineEdit("1e-4")
        self.ed_max_newton = QtWidgets.QLineEdit("50")

        fila_newton.addWidget(QtWidgets.QLabel("x0 inicial:"))
        fila_newton.addWidget(self.ed_x0_newton)
        fila_newton.addSpacing(10)
        fila_newton.addWidget(QtWidgets.QLabel("Tolerancia:"))
        fila_newton.addWidget(self.ed_tol_newton)
        fila_newton.addSpacing(10)
        fila_newton.addWidget(QtWidgets.QLabel("Máx. iteraciones:"))
        fila_newton.addWidget(self.ed_max_newton)
        v_newton.addLayout(fila_newton)

        self.btn_newton = btn("Calcular Newton-Raphson")
        self.btn_newton.clicked.connect(self.on_calcular_newton)
        v_newton.addWidget(self.btn_newton)

        # --- NUEVO BOTÓN NEWTON ---
        self.btn_paso_newton = btn("Ver Paso a Paso (Detallado)", kind="ghost")
        self.btn_paso_newton.clicked.connect(self.on_mostrar_detalles_newton)
        self.btn_paso_newton.setEnabled(False)
        v_newton.addWidget(self.btn_paso_newton)
        # --------------------------

        self.out_newton = OutputArea()
        v_newton.addWidget(self.out_newton)

        lay_newton.addWidget(grp_newton)
        lay_newton.addStretch(1)

        self.tabs.addTab(tab_newton, "Newton-Raphson")

        # --------------------------------------------------
        # Tab 3: Secante (método abierto)
        # --------------------------------------------------
        tab_secante = QtWidgets.QWidget()
        lay_secante = QtWidgets.QVBoxLayout(tab_secante)

        grp_sec = QtWidgets.QGroupBox("Método abierto: Secante")
        v_sec = QtWidgets.QVBoxLayout(grp_sec)

        fila_fx_s = QtWidgets.QHBoxLayout()
        self.ed_expr_secante = QtWidgets.QLineEdit("x^3 - x - 1")
        fila_fx_s.addWidget(QtWidgets.QLabel("f(x) ="))
        fila_fx_s.addWidget(self.ed_expr_secante)
        v_sec.addLayout(fila_fx_s)

        fila_sec1 = QtWidgets.QHBoxLayout()
        self.ed_x0_secante = QtWidgets.QLineEdit("1")
        self.ed_x1_secante = QtWidgets.QLineEdit("2")
        fila_sec1.addWidget(QtWidgets.QLabel("x0:"))
        fila_sec1.addWidget(self.ed_x0_secante)
        fila_sec1.addSpacing(10)
        fila_sec1.addWidget(QtWidgets.QLabel("x1:"))
        fila_sec1.addWidget(self.ed_x1_secante)
        v_sec.addLayout(fila_sec1)

        fila_sec2 = QtWidgets.QHBoxLayout()
        self.ed_tol_secante = QtWidgets.QLineEdit("1e-4")
        self.ed_max_secante = QtWidgets.QLineEdit("50")
        fila_sec2.addWidget(QtWidgets.QLabel("Tolerancia:"))
        fila_sec2.addWidget(self.ed_tol_secante)
        fila_sec2.addSpacing(10)
        fila_sec2.addWidget(QtWidgets.QLabel("Máx. iteraciones:"))
        fila_sec2.addWidget(self.ed_max_secante)
        v_sec.addLayout(fila_sec2)

        self.btn_secante = btn("Calcular Secante")
        self.btn_secante.clicked.connect(self.on_calcular_secante)
        v_sec.addWidget(self.btn_secante)

        # --- NUEVO BOTÓN SECANTE ---
        self.btn_paso_secante = btn("Ver Paso a Paso (Detallado)", kind="ghost")
        self.btn_paso_secante.clicked.connect(self.on_mostrar_detalles_secante)
        self.btn_paso_secante.setEnabled(False)
        v_sec.addWidget(self.btn_paso_secante)
        # ---------------------------

        self.out_secante = OutputArea()
        v_sec.addWidget(self.out_secante)

        lay_secante.addWidget(grp_sec)
        lay_secante.addStretch(1)

        self.tabs.addTab(tab_secante, "Secante")

        # --------------------------------------------------
        # Tab 4: Errores numéricos y propagación
        # --------------------------------------------------
        tab_errores = QtWidgets.QWidget()
        lay_errores = QtWidgets.QVBoxLayout(tab_errores)

        grp_err = QtWidgets.QGroupBox("Errores numéricos: error absoluto, relativo y relativo porcentual")
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
        self.btn_tipos_err = btn("Tipos de error (teoría)", kind="ghost")
        self.btn_tipos_err.clicked.connect(self.on_tipos_error)
        self.btn_ej_flot = btn("Ejemplo 0.1 + 0.2", kind="ghost")
        self.btn_ej_flot.clicked.connect(self.on_ejemplo_flotante)
        fila_btn_err.addWidget(self.btn_calc_err)
        fila_btn_err.addWidget(self.btn_tipos_err)
        fila_btn_err.addWidget(self.btn_ej_flot)
        fila_btn_err.addStretch(1)
        v_err.addLayout(fila_btn_err)

        self.out_err = OutputArea()
        v_err.addWidget(self.out_err)

        lay_errores.addWidget(grp_err)

        grp_prop = QtWidgets.QGroupBox("Propagación del error en una función y = f(x)")
        v_prop = QtWidgets.QVBoxLayout(grp_prop)

        fila_fx_prop = QtWidgets.QHBoxLayout()
        self.ed_fx_prop = QtWidgets.QLineEdit("x^2 + 3x")
        fila_fx_prop.addWidget(QtWidgets.QLabel("f(x) ="))
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

        lay_errores.addWidget(grp_prop)
        lay_errores.addStretch(1)

        self.tabs.addTab(tab_errores, "Errores y propagación")

        # --------------------------------------------------
        # Tab 5: Notación posicional
        # --------------------------------------------------
        tab_notacion = QtWidgets.QWidget()
        lay_not = QtWidgets.QVBoxLayout(tab_notacion)

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

        lay_not.addWidget(grp_not)
        lay_not.addStretch(1)

        self.tabs.addTab(tab_notacion, "Notación posicional")

    # ==================================================
    # Helpers para formatear tabla
    # ==================================================
    def _tabla_pasos(self, info):
        """
        Formatea la tabla de iteraciones para Bisección / Regla Falsa /
        Newton-Raphson / Secante.
        """
        pasos = info.get("pasos", [])
        if not pasos:
            return "No se generaron iteraciones."

        # Detectar tipo para ajustar header
        if info["metodo"] == "Newton-Raphson":
             header = (
                "Iter   xi         f(xi)        f'(xi)       ea           er%\n"
                "-------------------------------------------------------------------"
            )
        elif info["metodo"] == "Secante":
             header = (
                "Iter   xi_ant     xi         f(xi)        ea           er%\n"
                "-------------------------------------------------------------------"
            )
        else:
            header = (
                "Iter   a          b          xr         f(xr)        ea           er%\n"
                "-------------------------------------------------------------------------------"
            )
        filas = [header]

        for p in pasos:
            it = p["numero_de_iteracion"]
            
            # Helper para formatear
            def fval(v):
                if v is None: return "   ---   "
                return fmt_number(float(v), DEFAULT_DEC, False).rjust(10)

            ea = fval(p.get("error_absoluto_ea"))
            erp = fval(p.get("error_relativo_porcentual"))

            if info["metodo"] == "Newton-Raphson":
                xi = fval(p.get("xi"))
                fxi = fval(p.get("f_xi"))
                dfxi = fval(p.get("df_xi"))
                fila = f"{it:>3}  {xi} {fxi} {dfxi} {ea} {erp}"
            elif info["metodo"] == "Secante":
                xant = fval(p.get("xi_ant"))
                xi = fval(p.get("xi"))
                fxi = fval(p.get("f_xi"))
                fila = f"{it:>3}  {xant} {xi} {fxi} {ea} {erp}"
            else:
                a = fval(p.get("extremo_izquierdo_a"))
                b = fval(p.get("extremo_derecho_b"))
                xr = fval(p.get("aproximacion_actual_xr"))
                fxr = fval(p.get("valor_de_la_funcion_en_xr"))
                fila = f"{it:>3}  {a} {b} {xr} {fxr} {ea} {erp}"
            
            filas.append(fila)

        return "\n".join(filas)

    # ==================================================
    # Handlers Métodos cerrados
    # ==================================================
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

            # --- GUARDAR RESULTADO Y ACTIVAR BOTÓN ---
            self.info_raices_cache = info
            self.btn_paso_a_paso.setEnabled(True)
            # -----------------------------------------

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

            pasos = info.get("pasos", [])
            if pasos:
                ultimo = pasos[-1]
                iter_tot = info.get("iteraciones_totales", len(pasos))
                erp_final = ultimo.get("error_relativo_porcentual")
                ea_final = ultimo.get("error_absoluto_ea")

                lineas.append("Iteraciones totales: " + str(iter_tot))

                if erp_final is None:
                    erp_str = "---"
                else:
                    erp_str = fmt_number(erp_final, DEFAULT_DEC, False)

                lineas.append("Error relativo porcentual final er% = " + erp_str)

                criterio = info.get("criterio_de_paro")
                if criterio == "tolerancia" and ea_final is not None and ea_final <= tol:
                    comentario = (
                        "Comentario: el método alcanzó la tolerancia indicada; "
                        "la sucesión de aproximaciones se considera convergente "
                        "en el intervalo dado."
                    )
                elif criterio == "max_iter":
                    comentario = (
                        "Comentario: se alcanzó el número máximo de iteraciones "
                        "antes de cumplir la tolerancia; la convergencia aún no "
                        "queda garantizada con este criterio."
                    )
                else:
                    comentario = (
                        "Comentario: el proceso de iteraciones se detuvo según "
                        "el criterio de paro configurado."
                    )
                lineas.append(comentario)

            lineas.append("")
            lineas.append("Recordatorio de las fórmulas de error entre iteraciones:")
            lineas.append("ea = |xr - xr_anterior|")
            lineas.append("er = ea / |xr|")
            lineas.append("er% = er * 100")

            self.out_raices.clear_and_write("\n".join(lineas))
        except Exception as e:
            mensaje = str(e)
            if "El intervalo no es válido" in mensaje:
                self.out_raices.clear_and_write("El intervalo no es válido")
            else:
                self.out_raices.clear_and_write(f"Error: {mensaje}")

    def on_mostrar_detalles(self):
        """Muestra paso a paso para métodos cerrados"""
        if not self.info_raices_cache: return
        self._mostrar_dialogo(generar_reporte_paso_a_paso(self.info_raices_cache), self.info_raices_cache["metodo"])

    def on_mostrar_detalles_newton(self):
        """Muestra paso a paso para Newton"""
        if not self.info_newton_cache: return
        self._mostrar_dialogo(generar_reporte_abierto(self.info_newton_cache), "Newton-Raphson")

    def on_mostrar_detalles_secante(self):
        """Muestra paso a paso para Secante"""
        if not self.info_secante_cache: return
        self._mostrar_dialogo(generar_reporte_abierto(self.info_secante_cache), "Secante")

    def _mostrar_dialogo(self, texto, titulo):
        try:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f"Paso a Paso Detallado - {titulo}")
            dialog.resize(700, 700)
            layout = QtWidgets.QVBoxLayout(dialog)
            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Consolas", 10))
            text_edit.setPlainText(texto)
            layout.addWidget(text_edit)
            btn_ok = QtWidgets.QPushButton("Cerrar")
            btn_ok.clicked.connect(dialog.accept)
            layout.addWidget(btn_ok)
            dialog.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al generar reporte:\n{e}")

    # ==================================================
    # Handlers Newton-Raphson
    # ==================================================
    def on_calcular_newton(self):
        try:
            expr = self.ed_expr_newton.text().strip()
            x0 = float(self.ed_x0_newton.text())
            tol = float(self.ed_tol_newton.text())
            max_iter = int(self.ed_max_newton.text())

            info = newton_raphson_descriptiva(expr, x0, tol, max_iter)
            
            # GUARDAR CACHE Y ACTIVAR BOTON
            self.info_newton_cache = info
            self.btn_paso_newton.setEnabled(True)

            lineas = []
            lineas.append(f"Método: {info['metodo']}")
            lineas.append(f"f(x) = {expr}")
            lineas.append(f"x0 inicial: {x0}")
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

            pasos = info.get("pasos", [])
            if pasos:
                ultimo = pasos[-1]
                iter_tot = info.get("iteraciones_totales", len(pasos))
                erp_final = ultimo.get("error_relativo_porcentual")
                ea_final = ultimo.get("error_absoluto_ea")

                lineas.append("Iteraciones totales: " + str(iter_tot))

                erp_str = "---" if erp_final is None else fmt_number(erp_final, DEFAULT_DEC, False)
                lineas.append("Error relativo porcentual final er% = " + erp_str)

                criterio = info.get("criterio_de_paro")
                if criterio == "tolerancia" and ea_final is not None and ea_final <= tol:
                    comentario = (
                        "Comentario: el método alcanzó la tolerancia indicada; "
                        "la sucesión de aproximaciones se considera convergente "
                        "cerca de la raíz buscada."
                    )
                elif criterio == "max_iter":
                    comentario = (
                        "Comentario: se alcanzó el número máximo de iteraciones "
                        "antes de cumplir la tolerancia; puede requerirse más iteraciones "
                        "o un x0 diferente."
                    )
                else:
                    comentario = f"Comentario: criterio de paro '{criterio}'."
                lineas.append(comentario)

            self.out_newton.clear_and_write("\n".join(lineas))

        except Exception as e:
            self.out_newton.clear_and_write(f"Error: {e}")

    # ==================================================
    # Handlers Secante
    # ==================================================
    def on_calcular_secante(self):
        try:
            expr = self.ed_expr_secante.text().strip()
            x0 = float(self.ed_x0_secante.text())
            x1 = float(self.ed_x1_secante.text())
            tol = float(self.ed_tol_secante.text())
            max_iter = int(self.ed_max_secante.text())

            info = secante_descriptiva(expr, x0, x1, tol, max_iter)
            
            # GUARDAR CACHE Y ACTIVAR BOTON
            self.info_secante_cache = info
            self.btn_paso_secante.setEnabled(True)

            lineas = []
            lineas.append(f"Método: {info['metodo']}")
            lineas.append(f"f(x) = {expr}")
            lineas.append(f"x0 = {x0}, x1 = {x1}")
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

            pasos = info.get("pasos", [])
            if pasos:
                ultimo = pasos[-1]
                iter_tot = info.get("iteraciones_totales", len(pasos))
                erp_final = ultimo.get("error_relativo_porcentual")
                ea_final = ultimo.get("error_absoluto_ea")

                lineas.append("Iteraciones totales: " + str(iter_tot))

                erp_str = "---" if erp_final is None else fmt_number(erp_final, DEFAULT_DEC, False)
                lineas.append("Error relativo porcentual final er% = " + erp_str)

                criterio = info.get("criterio_de_paro")
                if criterio == "tolerancia" and ea_final is not None and ea_final <= tol:
                    comentario = (
                        "Comentario: el método alcanzó la tolerancia indicada; "
                        "la sucesión de aproximaciones se considera convergente."
                    )
                elif criterio == "max_iter":
                    comentario = (
                        "Comentario: se alcanzó el número máximo de iteraciones "
                        "antes de cumplir la tolerancia; puede requerir mejor intervalo inicial."
                    )
                else:
                    comentario = f"Comentario: criterio de paro '{criterio}'."
                lineas.append(comentario)

            self.out_secante.clear_and_write("\n".join(lineas))

        except Exception as e:
            self.out_secante.clear_and_write(f"Error: {e}")

    # ==================================================
    # Handlers Errores / Notación / Propagación
    # ==================================================
    def on_calcular_errores(self):
        try:
            m = float(self.ed_val_real.text())
            x_val = float(self.ed_val_aprox.text())
            info = calcular_errores(m, x_val)

            dec_err = 6
            ea = fmt_number(info["error_absoluto"], dec_err, False)
            er = (
                "---"
                if info["error_relativo"] is None
                else fmt_number(info["error_relativo"], dec_err, False)
            )
            erp = (
                "---"
                if info["error_relativo_porcentual"] is None
                else fmt_number(info["error_relativo_porcentual"], dec_err, False)
            )

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
