from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

from utilidad import *


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
    Normaliza la expresión de entrada:
    - Reemplaza ^ por ** (potencias estilo Python)
    - Inserta multiplicaciones explícitas (3x -> 3*x, x( -> x*()
    - Quita espacios sobrantes
    """
    expr = expr.replace("^", "**")
    expr = _insertar_multiplicacion_implicita(expr)
    return expr.strip()


def _make_func(expr: str):
    """
    Construye una función f(x) a partir de la cadena ingresada.

    1) Normaliza la expresión (x^3 - x - 1 -> x**3-x-1, 3x -> 3*x).
    2) Cuando se llama f(x0), se reemplaza cada 'x' por (x0) y
       se evalúa usando evaluar_expresion (sin usar argumento x=).
    """
    expr = _normalizar_expresion(str(expr))

    def f(x: float) -> float:
        expr_num = expr.replace("x", f"({x})")
        return float(evaluar_expresion(expr_num, exacto=False))

    return f



def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "---"
    return fmt_number(float(x), DEFAULT_DEC, False)


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
    """
    Método de Bisección con salida detallada para la GUI.

    Devuelve:
      - metodo
      - expresion
      - intervalo_inicial
      - tolerancia
      - pasos  (lista de dicts con a, b, xr, f(xr), ea, er, er%)
      - raiz_aproximada
      - valor_funcion_en_raiz
      - iteraciones_totales
      - error_relativo_porcentual_final
      - criterio_de_paro (tolerancia / max_iter / sin_iteraciones)
    """
    f = _make_func(expresion)
    a = float(a_inicial)
    b = float(b_inicial)
    tol = float(tolerancia)

    fa, fb = f(a), f(b)
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

        pasos.append(
            {
                "numero_de_iteracion": it,
                "extremo_izquierdo_a": a,
                "extremo_derecho_b": b,
                "aproximacion_actual_xr": xr,
                "valor_de_la_funcion_en_xr": fxr,
                "error_absoluto_ea": ea,
                "error_relativo": er,
                "error_relativo_porcentual": er_porcentual,
            }
        )

        # criterio de paro por tolerancia (error absoluto entre aproximaciones)
        if ea is not None and ea < tol:
            break

        # actualización de intervalo
        if fa * fxr < 0:
            b = xr
            fb = fxr
        else:
            a = xr
            fa = fxr

        xr_anterior = xr

    # Resumen de iteraciones y error relativo porcentual final
    if pasos:
        ultimo_paso = pasos[-1]
        ea_final = ultimo_paso["error_absoluto_ea"]
        iteraciones_totales = len(pasos)
        error_relativo_porcentual_final = ultimo_paso["error_relativo_porcentual"]
        if ea_final is not None and ea_final < tol:
            criterio_de_paro = "tolerancia"
        else:
            criterio_de_paro = "max_iter"
    else:
        iteraciones_totales = 0
        error_relativo_porcentual_final = None
        criterio_de_paro = "sin_iteraciones"

    raiz = xr
    valor_en_raiz = f(raiz)

    return {
        "metodo": "Bisección",
        "expresion": expresion,
        "intervalo_inicial": (a_inicial, b_inicial),
        "tolerancia": tolerancia,
        "pasos": pasos,
        "raiz_aproximada": raiz,
        "valor_funcion_en_raiz": valor_en_raiz,
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
    """
    Método de Regla Falsa (falsa posición) con salida detallada.

    Misma estructura de salida que biseccion_descriptiva.
    """
    f = _make_func(expresion)
    a = float(a_inicial)
    b = float(b_inicial)
    tol = float(tolerancia)

    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            "El intervalo no es válido. Se requiere f(a)·f(b) < 0 "
            "(la función debe cambiar de signo en [a,b])."
        )

    pasos: List[Dict[str, Any]] = []
    xr_anterior: Optional[float] = None
    xr: float = b

    for it in range(1, max_iter + 1):
        # fórmula de regla falsa (punto de intersección con el eje x)
        xr = b - fb * (b - a) / (fb - fa)
        fxr = f(xr)

        if xr_anterior is None:
            ea = None
            er = None
            er_porcentual = None
        else:
            ea = abs(xr - xr_anterior)
            er = ea / abs(xr) if xr != 0 else None
            er_porcentual = er * 100.0 if er is not None else None

        pasos.append(
            {
                "numero_de_iteracion": it,
                "extremo_izquierdo_a": a,
                "extremo_derecho_b": b,
                "aproximacion_actual_xr": xr,
                "valor_de_la_funcion_en_xr": fxr,
                "error_absoluto_ea": ea,
                "error_relativo": er,
                "error_relativo_porcentual": er_porcentual,
            }
        )

        if ea is not None and ea < tol:
            break

        if fa * fxr < 0:
            b = xr
            fb = fxr
        else:
            a = xr
            fa = fxr

        xr_anterior = xr

    # Resumen de iteraciones y error relativo porcentual final
    if pasos:
        ultimo_paso = pasos[-1]
        ea_final = ultimo_paso["error_absoluto_ea"]
        iteraciones_totales = len(pasos)
        error_relativo_porcentual_final = ultimo_paso["error_relativo_porcentual"]
        if ea_final is not None and ea_final < tol:
            criterio_de_paro = "tolerancia"
        else:
            criterio_de_paro = "max_iter"
    else:
        iteraciones_totales = 0
        error_relativo_porcentual_final = None
        criterio_de_paro = "sin_iteraciones"

    raiz = xr
    valor_en_raiz = f(raiz)

    return {
        "metodo": "Regla falsa",
        "expresion": expresion,
        "intervalo_inicial": (a_inicial, b_inicial),
        "tolerancia": tolerancia,
        "pasos": pasos,
        "raiz_aproximada": raiz,
        "valor_funcion_en_raiz": valor_en_raiz,
        "iteraciones_totales": iteraciones_totales,
        "error_relativo_porcentual_final": error_relativo_porcentual_final,
        "criterio_de_paro": criterio_de_paro,
    }


# Funciones envoltorio para cumplir con la especificación del programa
# Permiten invocar simplemente biseccion(...) y regla_falsa(...)
# usando la misma lógica implementada en las versiones descriptivas.

def biseccion(
    expresion: str,
    a_inicial: float,
    b_inicial: float,
    tolerancia: float,
    max_iter: int = 50,
) -> Dict[str, Any]:
    """
    Método de Bisección básico.
    Envoltorio sobre biseccion_descriptiva para facilitar su uso
    fuera de la interfaz gráfica.
    """
    return biseccion_descriptiva(expresion, a_inicial, b_inicial, tolerancia, max_iter)


def regla_falsa(
    expresion: str,
    a_inicial: float,
    b_inicial: float,
    tolerancia: float,
    max_iter: int = 50,
) -> Dict[str, Any]:
    """
    Método de Regla Falsa básico.
    Envoltorio sobre regla_falsa_descriptiva para facilitar su uso
    fuera de la interfaz gráfica.
    """
    return regla_falsa_descriptiva(expresion, a_inicial, b_inicial, tolerancia, max_iter)


# ======================================================
#  3. Cálculo de errores (ejercicio principal)
# ======================================================

def calcular_errores(valor_real: float, valor_aproximado: float) -> Dict[str, Any]:
    """
    Calcula error absoluto, relativo y relativo porcentual.
    """
    m = float(valor_real)
    x = float(valor_aproximado)
    ea = abs(m - x)
    er = ea / abs(m) if m != 0 else None
    er_porcentual = er * 100.0 if er is not None else None
    return {
        "valor_real": m,
        "valor_aproximado": x,
        "error_absoluto": ea,
        "error_relativo": er,
        "error_relativo_porcentual": er_porcentual,
    }


# ======================================================
#  4. Texto de tipos de error (para el documento)
# ======================================================

def tipos_de_error_texto() -> str:
    """
    Devuelve una explicación en texto de los principales tipos de error
    que pide la guía del Programa 8.
    """
    return (
        "Tipos de error en métodos numéricos:\n"
        "1) Error inherente:\n"
        "   Diferencia entre el valor real de una magnitud y el valor que\n"
        "   se toma como 'verdadero' en el modelo. A veces el dato de entrada\n"
        "   ya viene con error (por ejemplo, una medición experimental).\n\n"
        "2) Error de redondeo:\n"
        "   Se produce al limitar la cantidad de dígitos significativos que\n"
        "   se almacenan o muestran. Ejemplo: representar π como 3.14.\n\n"
        "3) Error de truncamiento:\n"
        "   Aparece cuando se reemplaza un proceso infinito (una serie, una\n"
        "   integral, una derivada exacta) por una versión finita o una\n"
        "   aproximación. Ejemplo: cortar una serie de Taylor después de\n"
        "   pocos términos.\n\n"
        "4) Overflow y underflow:\n"
        "   Overflow: el resultado es tan grande que sobrepasa el máximo que\n"
        "   puede representar el tipo de dato (por ejemplo, 1e308 en doble\n"
        "   precisión). Underflow: el número es tan pequeño que se aproxima a\n"
        "   cero y se pierde precisión.\n\n"
        "5) Error del modelo matemático:\n"
        "   Ocurre cuando el modelo que usamos para describir un fenómeno no\n"
        "   coincide exactamente con la realidad. Aunque resolvamos el modelo\n"
        "   de forma 'perfecta', la solución puede no representar bien al\n"
        "   sistema real.\n"
    )


# ======================================================
#  5. Ejemplo de punto flotante (0.1 + 0.2)
# ======================================================

def ejemplo_punto_flotante_texto() -> str:
    """
    Ejemplo obligatorio: 0.1 + 0.2. Explica por qué no es exacto.
    """
    return (
        "Ejemplo de punto flotante:\n"
        "En Python podemos ejecutar:\n"
        "    print(0.1 + 0.2)\n"
        "El resultado suele ser 0.30000000000000004 y NO 0.3 exacto.\n\n"
        "Esto ocurre porque 0.1 y 0.2 no tienen representación finita en\n"
        "base 2. La computadora almacena aproximaciones binarias de estos\n"
        "números y, al sumarlas, aparece el error de redondeo acumulado.\n"
        "Por eso, los números reales no siempre se representan exactamente\n"
        "en la máquina.\n"
    )


# ======================================================
#  6. Propagación del error en y = f(x)
# ======================================================

def propagacion_error(
    expresion: str,
    x0: float,
    delta_x: float,
) -> Dict[str, Any]:
    """
    Estima la propagación del error de x hacia y = f(x) usando la idea de
    derivada aproximada:

        Δy ≈ f(x0 + Δx) − f(x0)

    sin usar NumPy ni SciPy.
    """
    try:
        f = _make_func(expresion)
        x0 = float(x0)
        dx = float(delta_x)

        y0 = f(x0)
        y1 = f(x0 + dx)
        dy = y1 - y0

        ea_y = abs(dy)
        er_y = ea_y / abs(y0) if y0 != 0 else None
        erp_y = er_y * 100.0 if er_y is not None else None

        return {
            "x0": x0,
            "delta_x": dx,
            "y0": y0,
            "y1": y1,
            "delta_y": dy,
            "error_absoluto_y": ea_y,
            "error_relativo_y": er_y,
            "error_relativo_porcentual_y": erp_y,
        }
    except Exception:
        # Si algo falla al evaluar la función, lanzamos un error claro
        raise ValueError("No se pudo calcular la propagación del error. Verifique f(x), x0 y Δx.")

def generar_reporte_paso_a_paso(info: Dict[str, Any]) -> str:
    """
    Genera un reporte textual detallado imitando el estilo de pizarra/diapositiva
    basado en los pasos calculados previamente.
    """
    pasos = info.get("pasos", [])
    expr_original = info.get("expresion", "f(x)")
    metodo = info.get("metodo", "")
    
    expr_visual = expr_original.replace("**", "^")

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
        
        sub_xl = expr_visual.replace("x", f"({s_xl})")
        sub_xu = expr_visual.replace("x", f"({s_xu})")
        sub_xr = expr_visual.replace("x", f"({s_xr})")
        
        txt.append(f"\nITERACIÓN {it}:")
        txt.append(f"--------------------------------------------------")
        
        txt.append(f"   xi = {s_xl}      xu = {s_xu}")
        
        if "Bisección" in metodo:
            txt.append(f"   xr = (xi + xu) / 2  -->  xr = ({s_xl} + {s_xu}) / 2 = {s_xr}")
        else: # Regla Falsa

            txt.append(f"   xr (Falsa Posición) = {s_xr}")

        txt.append(f"   Ea = {s_ea}\n")

        func_eval = _make_func(expr_original)
        v_i = func_eval(xl)
        v_u = func_eval(xu)
        v_r = fxr # Este ya viene calculado

        txt.append(f"   vi = f(xi) = f({s_xl}) = {sub_xl} = {fmt_number(v_i, 5)}")
        txt.append(f"   vu = f(xu) = f({s_xu}) = {sub_xu} = {fmt_number(v_u, 5)}")
        txt.append(f"   vr = f(xr) = f({s_xr}) = {sub_xr} = {fmt_number(v_r, 5)}\n")

        # 4. Lógica de cambio de signo
        signo_vi = "(+)" if v_i >= 0 else "(-)"
        signo_vr = "(+)" if v_r >= 0 else "(-)"
        
        txt.append(f"   Verificación de signos: f(xi) * f(xr)")
        txt.append(f"   -> {fmt_number(v_i, 5)} * {fmt_number(v_r, 5)}")
        txt.append(f"   -> {signo_vi} * {signo_vr} {'<' if v_i*v_r < 0 else '>'} 0")
        
        if v_i * v_r < 0:
            txt.append(f"   >>> La raíz está entre [xi, xr] -> [{s_xl}, {s_xr}]")
        else:
            txt.append(f"   >>> La raíz está entre [xr, xu] -> [{s_xr}, {s_xu}]")
        
        txt.append("\n")

    return "\n".join(txt)