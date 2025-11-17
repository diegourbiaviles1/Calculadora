# metodosNumericos.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

from utilidad import evaluar_expresion, fmt_number, DEFAULT_DEC

# =========================
# 1) NOTACIÓN POSICIONAL
# =========================

def _separar_digitos_base10(num_str: str) -> Tuple[int, List[int]]:
    """
    Convierte el string en entero y devuelve sus dígitos en base 10.
    """
    s = num_str.strip().replace(" ", "")
    if not s or not (s.lstrip("+-").isdigit()):
        raise ValueError("Número inválido en base 10 (solo dígitos 0-9).")
    signo = -1 if s.startswith("-") else 1
    s_puro = s.lstrip("+-")
    n = int(s_puro) * signo
    digitos = [int(d) for d in s_puro]
    return n, digitos

def descomponer_base10(num_str: str, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Descompone un número en base 10, mostrando:
      - Producto de cada cifra por 10^posición.
      - Suma final.
    Ejemplo:
      84506 = 8×10^4 + 4×10^3 + 5×10^2 + 0×10^1 + 6×10^0
    """
    n, digitos = _separar_digitos_base10(num_str)
    pasos: List[str] = []

    pasos.append(f"Número dado (base 10): {n}")
    pasos.append(f"Escritura posicional en base 10 (de izquierda a derecha):")

    k = len(digitos)
    terminos = []
    linea_detalle = []

    for idx, d in enumerate(digitos):
        pos = k - 1 - idx  # potencia de 10
        terminos.append(f"{d}×10^{pos}")
        linea_detalle.append(f"{d}·10^{pos}")

    pasos.append("  " + " + ".join(linea_detalle))

    # Cálculo numérico de cada término
    pasos.append("\nCálculo numérico de cada término:")
    valores = []
    for idx, d in enumerate(digitos):
        pos = k - 1 - idx
        val = d * (10 ** pos)
        valores.append(val)
        pasos.append(f"  {d} × 10^{pos} = {val}")

    suma = sum(valores)
    pasos.append("\nSuma de todos los términos:")
    pasos.append("  " + " + ".join(str(v) for v in valores) + f" = {suma}")

    pasos.append(f"\nConclusión: {n} = " + " + ".join(terminos))

    return {
        "pasos": pasos,
        "conclusion": f"{n} = " + " + ".join(terminos)
    }

# --- Base 2 ---

def _validar_binario(num_str: str) -> str:
    s = num_str.strip().replace(" ", "")
    if not s:
        raise ValueError("Número binario vacío.")
    if any(ch not in "01" for ch in s):
        raise ValueError("Número binario inválido (solo 0 y 1).")
    return s

def descomponer_base2(num_str: str, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Descompone un número binario usando potencias de 2 y muestra su valor en base 10.
    Ejemplo:
      1111001 = 1·2^6 + 1·2^5 + ... + 1·2^0
    """
    b_str = _validar_binario(num_str)
    pasos: List[str] = []

    pasos.append(f"Número dado (base 2): {b_str}")
    pasos.append("Escritura posicional en base 2 (de izquierda a derecha):")

    k = len(b_str)
    terminos = []
    linea_detalle = []

    for idx, ch in enumerate(b_str):
        bit = int(ch)
        pos = k - 1 - idx
        terminos.append(f"{bit}·2^{pos}")
        linea_detalle.append(f"{bit}·2^{pos}")

    pasos.append("  " + " + ".join(linea_detalle))

    pasos.append("\nCálculo numérico de cada término:")
    valores = []
    for idx, ch in enumerate(b_str):
        bit = int(ch)
        pos = k - 1 - idx
        val = bit * (2 ** pos)
        valores.append(val)
        pasos.append(f"  {bit} × 2^{pos} = {val}")

    decimal = sum(valores)
    pasos.append("\nSuma de todos los términos:")
    pasos.append("  " + " + ".join(str(v) for v in valores) + f" = {decimal}")

    pasos.append(f"\nConclusión: {b_str}_2 = {decimal}_10")

    return {
        "pasos": pasos,
        "conclusion": f"{b_str}_2 = {decimal}_10"
    }

# =========================
# 2) CÁLCULO DE ERRORES
# =========================

def calcular_errores(xv_str: str, xa_str: str, dec: int = 4) -> Dict[str, Any]:
    """
    Calcula error absoluto y error relativo:
      Ea = |xv - xa|
      Er = Ea / |xv|
    xv_str y xa_str se evalúan como expresiones (1/3, 0.25, etc.).
    """
    xv = float(evaluar_expresion(xv_str, exacto=False))
    xa = float(evaluar_expresion(xa_str, exacto=False))

    Ea = abs(xv - xa)
    if abs(xv) < 1e-15:
        Er = float('inf')
    else:
        Er = Ea / abs(xv)

    pasos: List[str] = []
    pasos.append("=== Cálculo de errores ===")
    pasos.append(f"Valor verdadero  xv = {fmt_number(xv, dec)}")
    pasos.append(f"Valor aproximado xa = {fmt_number(xa, dec)}")
    pasos.append("")
    pasos.append("1) Error absoluto")
    pasos.append(f"   Ea = |xv - xa| = |{fmt_number(xv, dec)} - {fmt_number(xa, dec)}|")
    pasos.append(f"      = {fmt_number(Ea, dec)}")
    pasos.append("")
    pasos.append("2) Error relativo (en valor absoluto)")
    if Er == float('inf'):
        pasos.append("   Como |xv| ≈ 0, el error relativo no está definido (división entre cero).")
    else:
        pasos.append(f"   Er = Ea / |xv| = {fmt_number(Ea, dec)} / |{fmt_number(xv, dec)}|")
        pasos.append(f"      = {fmt_number(Er, dec)}  (≈ {fmt_number(Er*100, dec)} %)")

    conclusion = (
        f"El error absoluto es Ea = {fmt_number(Ea, dec)}, "
        f"y el error relativo es Er = {fmt_number(Er, dec)}"
        + (f" (≈ {fmt_number(Er*100, dec)} %)." if Er != float('inf') else ".")
    )

    return {
        "pasos": pasos,
        "conclusion": conclusion,
        "Ea": Ea,
        "Er": Er,
        "xv": xv,
        "xa": xa
    }

# =========================
# 3) PROPAGACIÓN DE ERROR
#    f(x) = sin(x) + x²
# =========================

def propagar_error_fijo(x_str: str, dx_str: str, dec: int = 4) -> Dict[str, Any]:
    """
    Estima la propagación del error en la función:
        f(x) = sin(x) + x²
        f'(x) = cos(x) + 2x
    Interpretación:
      - x (entrada) ≈ valor verdadero.
      - Δx = dx_str: incertidumbre/error en x.
      - Se calcula x_a = x_v + Δx.
      - Se comparan f(x_v) y f(x_a) y se aproxima Δy ≈ f'(x_v)·Δx.
    """
    xv = float(evaluar_expresion(x_str, exacto=False))
    dx = float(evaluar_expresion(dx_str, exacto=False))
    xa = xv + dx

    # Función y derivada
    def f(x: float) -> float:
        return math.sin(x) + x**2

    def df(x: float) -> float:
        return math.cos(x) + 2*x

    yv = f(xv)
    ya = f(xa)

    Ea_x = abs(xv - xa)          # debería ser |dx|
    Er_x = Ea_x / abs(xv) if abs(xv) > 1e-15 else float('inf')

    Ea_y = abs(yv - ya)
    Er_y = Ea_y / abs(yv) if abs(yv) > 1e-15 else float('inf')

    dy_est = abs(df(xv) * dx)

    pasos: List[str] = []
    pasos.append("=== Propagación del error en f(x) = sin(x) + x² ===\n")
    pasos.append(f"Valor verdadero aproximado de x : x_v = {fmt_number(xv, dec)}")
    pasos.append(f"Incertidumbre en x              : Δx  = {fmt_number(dx, dec)}")
    pasos.append(f"Valor aproximado de x           : x_a = x_v + Δx = {fmt_number(xa, dec)}")
    pasos.append("")
    pasos.append("1) Evaluación de la función:")
    pasos.append(f"   f(x_v) = sin({fmt_number(xv, dec)}) + ({fmt_number(xv, dec)})² = {fmt_number(yv, dec)}")
    pasos.append(f"   f(x_a) = sin({fmt_number(xa, dec)}) + ({fmt_number(xa, dec)})² = {fmt_number(ya, dec)}")
    pasos.append("")
    pasos.append("2) Errores en la variable x:")
    pasos.append(f"   Ea_x = |x_v - x_a| = {fmt_number(Ea_x, dec)}")
    if Er_x == float('inf'):
        pasos.append("   Er_x no definido porque |x_v| ≈ 0.")
    else:
        pasos.append(f"   Er_x = Ea_x / |x_v| = {fmt_number(Er_x, dec)}  (≈ {fmt_number(Er_x*100, dec)} %)")
    pasos.append("")
    pasos.append("3) Errores en la función f(x):")
    pasos.append(f"   Ea_y = |f(x_v) - f(x_a)| = |{fmt_number(yv, dec)} - {fmt_number(ya, dec)}| = {fmt_number(Ea_y, dec)}")
    if Er_y == float('inf'):
        pasos.append("   Er_y no definido porque |f(x_v)| ≈ 0.")
    else:
        pasos.append(f"   Er_y = Ea_y / |f(x_v)| = {fmt_number(Er_y, dec)}  (≈ {fmt_number(Er_y*100, dec)} %)")
    pasos.append("")
    pasos.append("4) Estimación por derivada (propagación de error):")
    pasos.append(f"   f'(x) = cos(x) + 2x")
    pasos.append(f"   f'({fmt_number(xv, dec)}) = cos({fmt_number(xv, dec)}) + 2·{fmt_number(xv, dec)}")
    pasos.append(f"                 ≈ {fmt_number(df(xv), dec)}")
    pasos.append(f"   Δy ≈ |f'(x_v)|·|Δx| = {fmt_number(abs(df(xv)), dec)}·{fmt_number(abs(dx), dec)}")
    pasos.append(f"         ≈ {fmt_number(dy_est, dec)}")
    pasos.append("")
    pasos.append("Resumen (tipo tabla):")
    pasos.append("  Magnitud    Valor verdadero   Valor aprox.   Error absoluto   Error relativo")
    pasos.append(f"  x           {fmt_number(xv, dec):>8}          {fmt_number(xa, dec):>8}       "
                 f"{fmt_number(Ea_x, dec):>8}        "
                 f"{'N/D' if Er_x==float('inf') else fmt_number(Er_x, dec)}")
    pasos.append(f"  f(x)        {fmt_number(yv, dec):>8}          {fmt_number(ya, dec):>8}       "
                 f"{fmt_number(Ea_y, dec):>8}        "
                 f"{'N/D' if Er_y==float('inf') else fmt_number(Er_y, dec)}")

    conclusion = (
        "Un pequeño error en x (Δx) se propaga a la función f(x): "
        f"el error estimado en f(x) es Δy ≈ {fmt_number(dy_est, dec)}, "
        f"mientras que el error real |f(x_v) - f(x_a)| = {fmt_number(Ea_y, dec)}."
    )

    return {
        "pasos": pasos,
        "conclusion": conclusion,
        "x_v": xv,
        "x_a": xa,
        "Ea_x": Ea_x,
        "Er_x": Er_x,
        "f_xv": yv,
        "f_xa": ya,
        "Ea_y": Ea_y,
        "Er_y": Er_y,
        "dy_estimado": dy_est
    }

# =========================
# 4) CONCEPTOS + PUNTO FLOTANTE
# =========================

def conceptos_y_punto_flotante(dec: int = 10) -> Dict[str, Any]:
    """
    Devuelve:
      - Una lista de (título, descripción) con los tipos de error.
      - Una lista de líneas mostrando la "paradoja" 0.1 + 0.2 == 0.3
        y un pequeño taller guiado con sumas repetidas.
    """
    conceptos: List[Tuple[str, str]] = []

    conceptos.append((
        "Error inherente",
        "Es el error que trae el dato desde el mundo real. "
        "Por ejemplo, medir una distancia con una cinta que solo tiene milímetros."
    ))
    conceptos.append((
        "Error de redondeo",
        "Se produce al redondear números para ajustarlos al formato de la máquina. "
        "Ej: 1/3 ≈ 0.3333 al usar 4 decimales."
    ))
    conceptos.append((
        "Error de truncamiento",
        "Aparece al reemplazar un proceso infinito por uno finito. "
        "Por ejemplo, usar unas pocas iteraciones de una serie infinita."
    ))
    conceptos.append((
        "Overflow y underflow",
        "Overflow: el número es demasiado grande para representarse (se desborda). "
        "Underflow: el número es tan pequeño que se aproxima a 0 en la máquina."
    ))
    conceptos.append((
        "Error del modelo matemático",
        "Se origina cuando el modelo que usamos para describir un fenómeno real no es perfecto. "
        "Ej: suponer movimiento sin fricción cuando en realidad sí hay fricción."
    ))

    # Punto flotante: 0.1 + 0.2
    pf: List[str] = []
    pf.append("=== Ejemplo de punto flotante en Python ===")
    pf.append(">>> 0.1 + 0.2")
    pf.append(f"{0.1 + 0.2!r}")
    pf.append("")
    pf.append(">>> 0.1 + 0.2 == 0.3")
    pf.append(f"{(0.1 + 0.2 == 0.3)!r}")
    pf.append("")
    pf.append("La comparación da False porque 0.1 y 0.2 no se representan de forma exacta ")
    pf.append("en binario. Internamente, 0.1 + 0.2 es algo como 0.30000000000000004.")

    # Pequeño "taller guiado" sin NumPy: sumas repetidas
    pf.append("\n=== Taller guiado: sumas repetidas ===")
    pf.append("Sea s1 la suma exacta, y s2 la suma acumulando errores de redondeo:")
    n = 100
    s1 = 0.1 * n
    s2 = 0.0
    for _ in range(n):
        s2 += 0.1
    pf.append(f"Sumar 0.1 exactamente:   0.1 * {n} = {s1!r}")
    pf.append(f"Sumar 0.1 en un bucle:   s2 = {s2!r}")
    pf.append("")
    pf.append("La idea es mostrar que al acumular muchas veces un número que ya está ")
    pf.append("representado de forma aproximada, los errores de redondeo se acumulan poco a poco.")

    return {
        "conceptos": conceptos,
        "punto_flotante": pf
    }
