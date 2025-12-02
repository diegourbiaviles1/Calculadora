# metodos_numericos.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
import sympy as sp
from utilidad import *

# ======================================================
#  1. Funciones auxiliares para parseo y funciones
# ======================================================

def _insertar_multiplicacion_implicita(expr: str) -> str:
    """
    Inserta '*' donde suele haber multiplicación implícita:
    - 3x  -> 3*x
    - )x  -> )*x
    - x2  -> x*2
    - x(  -> x*(
    """
    # dígito o ) seguido de x
    expr = re.sub(r'(\d|\))\s*x', r'\1*x', expr)
    # x seguido de dígito o (
    expr = re.sub(r'x\s*(\d|\()', r'x*\1', expr)
    return expr

def _normalizar_expresion(expr: str) -> str:
    """
    Normaliza la expresión de entrada básica.
    """
    expr = expr.replace("^", "**")
    expr = _insertar_multiplicacion_implicita(expr)
    return expr.strip()

def _make_func(expr_str: str):
    """
    Construye una función f(x) usando sympy para soportar trigonométricas
    y otras funciones complejas de forma segura y rápida.
    """
    expr_str = expr_str.replace("^", "**")
    # Soporte básico para e^x -> exp(x) si el usuario lo escribe así
    expr_str = re.sub(r'e\*\*(\w+)', r'exp(\1)', expr_str)
    
    try:
        # Intentar parsear con sympy
        x_sym = sp.Symbol('x')
        # sympify convierte strings a expresiones sympy (soporta sin, cos, etc.)
        expr = sp.sympify(expr_str)
        # lambdify convierte la expresión sympy a una función python nativa (usando math)
        f = sp.lambdify(x_sym, expr, modules=["math"])
        return f
    except Exception:
        # Fallback al método anterior manual si sympy falla (raro)
        def f_manual(x: float) -> float:
            expr_num = expr_str.replace("x", f"({x})")
            return float(evaluar_expresion(expr_num, exacto=False))
        return f_manual

def _parse_funcion_sympy(expr_str: str):
    """
    Retorna la función numérica y la expresión simbólica de Sympy.
    Útil para métodos que requieren derivadas (Newton).
    """
    expr_str = expr_str.replace("^", "**")
    expr_str = re.sub(r'e\*\*(\w+)', r'exp(\1)', expr_str) # e^x -> exp(x)
    
    x_sym = sp.Symbol('x')
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x_sym, expr, modules=["math"])
    return f, expr

# ======================================================
#  2. Métodos de raíces: Bisección y Regla Falsa
# ======================================================

def biseccion_descriptiva(
    expresion: str,
    a_inicial: float,
    b_inicial: float,
    tolerancia: float,
    max_iter: int = 50,
) -> Dict[str, Any]:
    f = _make_func(expresion)
    a = float(a_inicial)
    b = float(b_inicial)
    tol = float(tolerancia)

    try:
        fa, fb = f(a), f(b)
    except Exception as e:
        raise ValueError(f"Error evaluando la función en los extremos: {e}")

    if fa * fb > 0:
        raise ValueError(
            "El intervalo no es válido. Se requiere f(a)·f(b) < 0 "
            "(la función debe cambiar de signo en [a,b])."
        )

    pasos: List[Dict[str, Any]] = []
    xr_anterior: Optional[float] = None
    xr: float = (a + b) / 2.0

    for it in range(1, max_iter + 1):
        xr = (a + b) / 2.0
        fxr = f(xr)

        if xr_anterior is None:
            ea = None
            er = None
            er_porcentual = None
        else:
            ea = abs(xr - xr_anterior)
            er = ea / abs(xr) if xr != 0 else None
            er_porcentual = er * 100.0 if er is not None else None

        pasos.append({
            "numero_de_iteracion": it,
            "extremo_izquierdo_a": a,
            "extremo_derecho_b": b,
            "aproximacion_actual_xr": xr,
            "valor_de_la_funcion_en_xr": fxr,
            "error_absoluto_ea": ea,
            "error_relativo": er,
            "error_relativo_porcentual": er_porcentual,
        })

        if ea is not None and ea < tol:
            break

        if fa * fxr < 0:
            b = xr
            fb = fxr
        else:
            a = xr
            fa = fxr
        xr_anterior = xr

    # Resumen
    if pasos:
        ultimo_paso = pasos[-1]
        ea_final = ultimo_paso["error_absoluto_ea"]
        iteraciones_totales = len(pasos)
        error_relativo_porcentual_final = ultimo_paso["error_relativo_porcentual"]
        criterio_de_paro = "tolerancia" if (ea_final is not None and ea_final < tol) else "max_iter"
    else:
        iteraciones_totales = 0
        error_relativo_porcentual_final = None
        criterio_de_paro = "sin_iteraciones"

    return {
        "metodo": "Bisección",
        "expresion": expresion,
        "intervalo_inicial": (a_inicial, b_inicial),
        "tolerancia": tolerancia,
        "pasos": pasos,
        "raiz_aproximada": xr,
        "valor_funcion_en_raiz": f(xr),
        "iteraciones_totales": iteraciones_totales,
        "error_relativo_porcentual_final": error_relativo_porcentual_final,
        "criterio_de_paro": criterio_de_paro,
    }

def regla_falsa_descriptiva(
    expresion: str,
    a_inicial: float,
    b_inicial: float,
    tolerancia: float,
    max_iter: int = 50,
) -> Dict[str, Any]:
    f = _make_func(expresion)
    a = float(a_inicial)
    b = float(b_inicial)
    tol = float(tolerancia)

    try:
        fa, fb = f(a), f(b)
    except Exception as e:
        raise ValueError(f"Error evaluando la función: {e}")

    if fa * fb > 0:
        raise ValueError(
            "El intervalo no es válido. Se requiere f(a)·f(b) < 0."
        )

    pasos: List[Dict[str, Any]] = []
    xr_anterior: Optional[float] = None
    xr: float = b

    for it in range(1, max_iter + 1):
        # fórmula de regla falsa
        denom = (fb - fa)
        if denom == 0: break # Evitar división por cero
        xr = b - fb * (b - a) / denom
        fxr = f(xr)

        if xr_anterior is None:
            ea = None
            er = None
            er_porcentual = None
        else:
            ea = abs(xr - xr_anterior)
            er = ea / abs(xr) if xr != 0 else None
            er_porcentual = er * 100.0 if er is not None else None

        pasos.append({
            "numero_de_iteracion": it,
            "extremo_izquierdo_a": a,
            "extremo_derecho_b": b,
            "aproximacion_actual_xr": xr,
            "valor_de_la_funcion_en_xr": fxr,
            "error_absoluto_ea": ea,
            "error_relativo": er,
            "error_relativo_porcentual": er_porcentual,
        })

        if ea is not None and ea < tol:
            break

        if fa * fxr < 0:
            b = xr
            fb = fxr
        else:
            a = xr
            fa = fxr
        xr_anterior = xr

    if pasos:
        ultimo_paso = pasos[-1]
        ea_final = ultimo_paso["error_absoluto_ea"]
        iteraciones_totales = len(pasos)
        error_relativo_porcentual_final = ultimo_paso["error_relativo_porcentual"]
        criterio_de_paro = "tolerancia" if (ea_final is not None and ea_final < tol) else "max_iter"
    else:
        iteraciones_totales = 0
        error_relativo_porcentual_final = None
        criterio_de_paro = "sin_iteraciones"

    return {
        "metodo": "Regla falsa",
        "expresion": expresion,
        "intervalo_inicial": (a_inicial, b_inicial),
        "tolerancia": tolerancia,
        "pasos": pasos,
        "raiz_aproximada": xr,
        "valor_funcion_en_raiz": f(xr),
        "iteraciones_totales": iteraciones_totales,
        "error_relativo_porcentual_final": error_relativo_porcentual_final,
        "criterio_de_paro": criterio_de_paro,
    }

# ======================================================
#  3. Métodos Abiertos: Newton y Secante
# ======================================================

def newton_raphson_descriptiva(expr_str: str, x0: float, tol: float, max_iter: int) -> Dict[str, Any]:
    f, expr_sym = _parse_funcion_sympy(expr_str)
    x_sym = sp.Symbol('x')
    df_expr = sp.diff(expr_sym, x_sym)
    df = sp.lambdify(x_sym, df_expr, modules=["math"])

    pasos = []
    xr_anterior = None
    xr = float(x0)
    criterio = "max_iter"
    ea = None

    # String de la derivada para el reporte
    df_str = str(df_expr).replace("**", "^")

    for it in range(1, max_iter + 1):
        try:
            fx = f(xr)
            dfx = df(xr)
        except Exception:
            criterio = "error_matematico"
            break

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
            "xi": xr,
            "f_xi": fx,
            "df_xi": dfx,
            "xi_nuevo": xr_nuevo,
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

    return {
        "metodo": "Newton-Raphson",
        "expresion": expr_str,
        "derivada_str": df_str,
        "raiz_aproximada": xr,
        "valor_funcion_en_raiz": f(xr),
        "iteraciones_totales": len(pasos),
        "criterio_de_paro": criterio,
        "pasos": pasos,
    }

def secante_descriptiva(expr_str: str, x0: float, x1: float, tol: float, max_iter: int) -> Dict[str, Any]:
    f = _make_func(expr_str)
    pasos = []
    xr_ant = float(x0)
    xr = float(x1)
    criterio = "max_iter"

    for it in range(1, max_iter + 1):
        try:
            f_xr = f(xr)
            f_xr_ant = f(xr_ant)
        except Exception:
            criterio = "error_matematico"
            break

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
            "xi_ant": xr_ant,
            "xi": xr,
            "f_xi_ant": f_xr_ant,
            "f_xi": f_xr,
            "xi_nuevo": xr_nuevo,
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

    return {
        "metodo": "Secante",
        "expresion": expr_str,
        "raiz_aproximada": xr,
        "valor_funcion_en_raiz": f(xr),
        "iteraciones_totales": len(pasos),
        "criterio_de_paro": criterio,
        "pasos": pasos,
    }

# ======================================================
#  4. Generadores de Reportes (con formato bonito)
# ======================================================

def generar_reporte_paso_a_paso(info: Dict[str, Any]) -> str:
    """Genera reporte para Bisección y Regla Falsa."""
    pasos = info.get("pasos", [])
    expr_original = info.get("expresion", "f(x)")
    metodo = info.get("metodo", "")
    
    # FORMATO VISUAL MEJORADO (superíndices)
    expr_visual = formatear_superindice(expr_original)

    txt = []
    txt.append(f"=== REPORTE DETALLADO: {metodo.upper()} ===\n")
    txt.append(f"Función: f(x) = {expr_visual}")
    txt.append("-" * 60)

    for p in pasos:
        it = p["numero_de_iteracion"]
        xl = p["extremo_izquierdo_a"] 
        xu = p["extremo_derecho_b"]   
        xr = p["aproximacion_actual_xr"]
        fxr = p["valor_de_la_funcion_en_xr"]
        ea = p["error_absoluto_ea"]
        
        s_xl = fmt_number(xl, 5)
        s_xu = fmt_number(xu, 5)
        s_xr = fmt_number(xr, 5)
        s_ea = fmt_number(ea, 5) if ea is not None else "---"
        
        # Sustitución visual (aproximada para referencia)
        sub_xl = expr_visual.replace("x", f"({s_xl})")
        sub_xu = expr_visual.replace("x", f"({s_xu})")
        sub_xr = expr_visual.replace("x", f"({s_xr})")
        
        txt.append(f"\nITERACIÓN {it}:")
        txt.append(f"--------------------------------------------------")
        txt.append(f"   xi = {s_xl}      xu = {s_xu}")
        
        if "Bisección" in metodo:
            txt.append(f"   xr = (xi + xu) / 2  -->  xr = ({s_xl} + {s_xu}) / 2 = {s_xr}")
        else: 
            txt.append(f"   xr (Falsa Posición) = {s_xr}")

        txt.append(f"   Ea = {s_ea}\n")

        # Recalcular valores para mostrar en el log visual
        f = _make_func(expr_original)
        try:
            v_i, v_u = f(xl), f(xu)
            v_r = fxr 

            txt.append(f"   f(xi) = {fmt_number(v_i, 5)}")
            txt.append(f"   f(xu) = {fmt_number(v_u, 5)}")
            txt.append(f"   f(xr) = {fmt_number(v_r, 5)}\n")

            signo_vi = "(+)" if v_i >= 0 else "(-)"
            signo_vr = "(+)" if v_r >= 0 else "(-)"
            
            txt.append(f"   Verificación de signos: f(xi) * f(xr)")
            txt.append(f"   -> {signo_vi} * {signo_vr} {'<' if v_i*v_r < 0 else '>'} 0")
            
            if v_i * v_r < 0:
                txt.append(f"   >>> La raíz está entre [xi, xr] -> [{s_xl}, {s_xr}]")
            else:
                txt.append(f"   >>> La raíz está entre [xr, xu] -> [{s_xr}, {s_xu}]")
        except:
            txt.append("   (Error evaluando función para detalles de signos)")
        
        txt.append("\n")

    return "\n".join(txt)

def generar_reporte_abierto(info: dict) -> str:
    """Genera reporte para Newton y Secante."""
    metodo = info["metodo"]
    # FORMATO VISUAL MEJORADO
    expr_visual = formatear_superindice(info["expresion"])
    pasos = info["pasos"]
    
    lines = []
    lines.append(f"=== REPORTE DETALLADO: {metodo.upper()} ===")
    lines.append(f"Función: f(x) = {expr_visual}")
    
    if metodo == "Newton-Raphson":
        # Formatear la derivada también
        derivada_vis = formatear_superindice(info['derivada_str'])
        lines.append(f"Derivada analítica: f'(x) = {derivada_vis}")
        lines.append("-" * 60)
        
        for p in pasos:
            i = p["numero_de_iteracion"]
            xi = p["xi"]
            f_xi, df_xi = p["f_xi"], p["df_xi"]
            res, ea = p["xi_nuevo"], p["error_absoluto_ea"]
            
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
                lines.append(f"   3. Error aproximado: Ea = |{s_res} - {s_xi}| ≈ {s_ea}")
            lines.append("-" * 40)

    elif metodo == "Secante":
        lines.append("-" * 60)
        for p in pasos:
            i = p["numero_de_iteracion"]
            xi_ant, xi = p["xi_ant"], p["xi"]
            fi_ant, fi = p["f_xi_ant"], p["f_xi"]
            res, ea = p["xi_nuevo"], p["error_absoluto_ea"]
            
            s_xia, s_xi = fmt_number(xi_ant, 6), fmt_number(xi, 6)
            s_fa, s_fi = fmt_number(fi_ant, 6), fmt_number(fi, 6)
            s_res = fmt_number(res, 6)
            s_ea  = fmt_number(ea, 7) if ea is not None else "---"
            
            lines.append(f"\nIteración {i} (usar x{i-2}={s_xia}, x{i-1}={s_xi}):")
            lines.append(f"   1. Evaluaciones: f({s_xia}) ≈ {s_fa}, f({s_xi}) ≈ {s_fi}")
            lines.append(f"   2. Fórmula de la secante:")
            lines.append(f"      x{i} = {s_xi} - [ ({s_fi}) * ({s_xi} - {s_xia}) / ({s_fi} - {s_fa}) ]")
            lines.append(f"         ≈ {s_res}")
            
            if ea is not None:
                lines.append(f"   Error aproximado: Ea = |{s_res} - {s_xi}| ≈ {s_ea}")
            lines.append("-" * 40)
            
    return "\n".join(lines)

# Envoltorios para compatibilidad
def biseccion(expr, a, b, tol, mx=50):
    return biseccion_descriptiva(expr, a, b, tol, mx)

def regla_falsa(expr, a, b, tol, mx=50):
    return regla_falsa_descriptiva(expr, a, b, tol, mx)

# ======================================================
#  Otros Cálculos (Errores, etc.) - Sin cambios lógicos
# ======================================================
def calcular_errores(valor_real: float, valor_aproximado: float) -> Dict[str, Any]:
    m, x = float(valor_real), float(valor_aproximado)
    ea = abs(m - x)
    er = ea / abs(m) if m != 0 else None
    return {
        "valor_real": m, "valor_aproximado": x,
        "error_absoluto": ea, "error_relativo": er,
        "error_relativo_porcentual": er * 100.0 if er is not None else None,
    }

def tipos_de_error_texto() -> str:
    return (
        "Tipos de error en métodos numéricos:\n"
        "1) Error inherente: Datos de entrada inexactos.\n"
        "2) Error de redondeo: Limitación de dígitos significativos.\n"
        "3) Error de truncamiento: Aproximar procesos infinitos.\n"
        "4) Overflow/Underflow: Números demasiado grandes o pequeños.\n"
        "5) Error del modelo: El modelo matemático no es perfecto."
    )

def ejemplo_punto_flotante_texto() -> str:
    return (
        "Ejemplo de punto flotante:\n"
        "    0.1 + 0.2 = 0.30000000000000004\n"
        "Esto ocurre por la representación binaria inexacta de 0.1 y 0.2."
    )

def propagacion_error(expresion: str, x0: float, delta_x: float) -> Dict[str, Any]:
    f = _make_func(expresion)
    x0, dx = float(x0), float(delta_x)
    y0 = f(x0)
    y1 = f(x0 + dx)
    dy = y1 - y0
    ea = abs(dy)
    return {
        "x0": x0, "delta_x": dx, "y0": y0, "y1": y1, "delta_y": dy,
        "error_absoluto_y": ea,
        "error_relativo_y": ea / abs(y0) if y0 != 0 else None,
        "error_relativo_porcentual_y": (ea / abs(y0) * 100) if y0 != 0 else None,
    }