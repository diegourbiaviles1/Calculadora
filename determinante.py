# determinante.py — versión con "paso a paso" detallado
from __future__ import annotations
from typing import List, Dict, Any
from utilidad import fmt_number, DEFAULT_DEC, format_matrix_bracket
from fractions import Fraction

Matriz = List[List[float]]

# ---------- Validaciones ----------
def _check_square(A: Matriz):
    if not A or not A[0]:
        raise ValueError("Matriz vacía.")
    n = len(A)
    m = len(A[0])
    if any(len(f) != m for f in A):
        raise ValueError("Matriz no rectangular.")
    if n != m:
        raise ValueError("La matriz debe ser cuadrada.")

def _minor(A: Matriz, i: int, j: int) -> Matriz:
    return [fila[:j] + fila[j+1:] for k, fila in enumerate(A) if k != i]

def _cof_sign(i: int, j: int) -> int:
    return -1 if ((i + j) % 2) else 1

def _mejor_fila_para_expandir(A: Matriz) -> int:
    """elige la fila con más ceros para que la expansión sea más clara/compacta."""
    zeros_por_fila = [sum(1 for x in fila if abs(x) < 1e-15) for fila in A]
    return max(range(len(A)), key=lambda r: zeros_por_fila[r])

def det_cofactores(
    A: Matriz,
    dec: int = DEFAULT_DEC,
    expand_index: int = 0,
    modo: str = "renglon"  # "renglon" o "columna"
) -> Dict[str, Any]:
    """
    Determinante por cofactores con formato compacto (pizarra):
      - Puede expandir por un renglón o una columna (modo).
      - Muestra mapa de signos.
      - Usa notación con barras |A|.
      - Formato de 4 líneas para claridad.
    """
    _check_square(A)
    n = len(A)
    if not (0 <= expand_index < n):
        raise ValueError("expand_index fuera de rango.")
    if modo not in ("renglon", "columna"):
        raise ValueError("modo debe ser 'renglon' o 'columna'.")

    pasos: List[str] = []

    # --- Helpers ---
    def _sgn(i: int, j: int) -> int:
        return 1 if ((i + j) % 2 == 0) else -1

    def _sym(s: int) -> str:
        return "+" if s > 0 else "−"

    def _bars_inline(M: Matriz, deci: int) -> str:
        filas = [" ".join(fmt_number(x, deci) for x in fila) for fila in M]
        return "| " + " ; ".join(filas) + " |"

    def _det_numeric(M: Matriz) -> float:
        k = len(M)
        if k == 1:
            return float(M[0][0])
        if k == 2:
            a, b = float(M[0][0]), float(M[0][1])
            c, d = float(M[1][0]), float(M[1][1])
            return a * d - b * c
        total = 0.0
        for j in range(k):
            aij = float(M[0][j])
            if abs(aij) < 1e-15:
                continue
            total += _sgn(0, j) * aij * _det_numeric(_minor(M, 0, j))
        return total

    def _minor_expr_or_value(M: Matriz, deci: int) -> str:
        if len(M) == 2:
            a, b = fmt_number(M[0][0], deci), fmt_number(M[0][1], deci)
            c, d = fmt_number(M[1][0], deci), fmt_number(M[1][1], deci)
            return f"{a}·{d} − {b}·{c}"
        return fmt_number(_det_numeric(M), deci)

    def _bars_block(M: Matriz, deci: int) -> str:
        if not M:
            return "| |"
        r, c = len(M), len(M[0])
        grid = [[fmt_number(x, deci, False) for x in fila] for fila in M]
        widths = [max(len(grid[i][j]) for i in range(r)) for j in range(c)]
        lines = []
        for i in range(r):
            cells = "  ".join(grid[i][j].rjust(widths[j]) for j in range(c))
            lines.append(f"| {cells} |")
        return "\n".join(lines)

    # --- Cabecera ---
    if modo == "renglon":
        pasos.append(f"[Cofactores]\nUsando el renglón {expand_index+1} (expansión por cofactores):")
    else:
        pasos.append(f"[Cofactores]\nUsando la columna {expand_index+1} (expansión por cofactores):")

    pasos.append("Matriz actual entre barras:")
    pasos.append(_bars_block(A, dec))

    # --- Mapa de signos ---
    pasos.append("Mapa de signos de cofactores (+/−):")
    for i in range(n):
        pasos.append("  ".join(_sym(_sgn(i, j)) for j in range(n)))

    # --- Seleccionar elementos según modo ---
    terms = []
    if modo == "renglon":
        for j in range(n):
            aij = float(A[expand_index][j])
            if abs(aij) < 1e-15:
                continue
            s = _sgn(expand_index, j)
            Mij = _minor(A, expand_index, j)
            detM = _det_numeric(Mij)
            terms.append({"s": s, "val": aij, "M": Mij, "detM": detM})
    else:  # columna
        for i in range(n):
            aij = float(A[i][expand_index])
            if abs(aij) < 1e-15:
                continue
            s = _sgn(i, expand_index)
            Mij = _minor(A, i, expand_index)
            detM = _det_numeric(Mij)
            terms.append({"s": s, "val": aij, "M": Mij, "detM": detM})

    if not terms:
        pasos.append("Todos los términos del renglón/columna son nulos.")
        return {"metodo": "cofactores", "det": 0.0, "pasos": pasos, "conclusion": "|A| = 0"}

    # --- Línea 1: expansión simbólica ---
    linea1 = []
    for k, t in enumerate(terms):
        sym = "" if (k == 0 and t["s"] > 0) else (" − " if t["s"] < 0 else " + ")
        linea1.append(f"{sym}{fmt_number(abs(t['val']), dec)}{_bars_inline(t['M'], dec)}")
    pasos.append("|A| = " + "".join(linea1))

    # --- Línea 2: menores como expresiones ---
    linea2 = []
    for k, t in enumerate(terms):
        sym = "" if (k == 0 and t["s"] > 0) else (" − " if t["s"] < 0 else " + ")
        linea2.append(f"{sym}{fmt_number(abs(t['val']), dec)}({_minor_expr_or_value(t['M'], dec)})")
    pasos.append("     = " + "".join(linea2))

    # --- Línea 3: sustituir por valores numéricos ---
    linea3 = []
    for k, t in enumerate(terms):
        sym = "" if (k == 0 and t["s"] > 0) else (" − " if t["s"] < 0 else " + ")
        linea3.append(f"{sym}{fmt_number(abs(t['val']), dec)}({fmt_number(t['detM'], dec)})")
    pasos.append("     = " + "".join(linea3))

    # --- Línea 4: resultados finales ---
    total = sum(t["s"] * t["val"] * t["detM"] for t in terms)
    productos = []
    for k, t in enumerate(terms):
        prod = t["s"] * t["val"] * t["detM"]
        sym = "" if (k == 0 and prod >= 0) else (" − " if prod < 0 else " + ")
        productos.append(f"{sym}{fmt_number(abs(prod), dec)}")
    pasos.append("     = " + "".join(productos) + f" = {fmt_number(total, dec)}")

    return {
        "metodo": "cofactores",
        "det": total,
        "pasos": pasos,
        "conclusion": f"|A| = {fmt_number(total, dec)}  (usando {'renglón' if modo=='renglon' else 'columna'} {expand_index+1})"
    }





# ---------- Regla de Sarrus (solo 3×3) con diagonales explícitas ----------
# ---------- Regla de Sarrus (3×3) con matriz extendida y diagonales explícitas ----------
def det_sarrus(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Presentación tipo pizarra:
      • Muestra |A|.
      • Muestra la matriz extendida (se repiten las dos primeras columnas a la derecha).
      • Lista y desarrolla cada diagonal ↘ (principal) y ↙ (secundaria).
      • Escribe el desarrollo: |A| = (suma ↘) − (suma ↙) = ...
    """
    _check_square(A)
    if len(A) != 3 or len(A[0]) != 3:
        raise ValueError("La regla de Sarrus solo aplica a matrices 3×3.")

    a = A  # alias corto
    pasos: List[str] = []

    def num(x): return fmt_number(x, dec)

    # 1) Matriz original entre barras
    pasos.append("[Sarrus]\n")
    pasos.append("Matriz A entre barras:\n" + format_matrix_bracket(a, dec))

    # 2) Matriz extendida (añadir las dos primeras columnas a la derecha)
    ext = [
        [a[0][0], a[0][1], a[0][2], a[0][0], a[0][1]],
        [a[1][0], a[1][1], a[1][2], a[1][0], a[1][1]],
        [a[2][0], a[2][1], a[2][2], a[2][0], a[2][1]],
    ]
    def format_matrix_bracket_wide(M):
        rows = []
        for fila in M:
            rows.append("[ " + "  ".join(fmt_number(x, dec) for x in fila) + " ]")
        return "\n".join(rows)

    pasos.append("\nMatriz extendida (se añaden las 2 primeras columnas a la derecha):\n" +
                 format_matrix_bracket_wide(ext))

    # 3) Diagonales ↘ (principales) y ↙ (secundarias)
    # Principales: a00·a11·a22, a01·a12·a20, a02·a10·a21
    d1 = a[0][0] * a[1][1] * a[2][2]
    d2 = a[0][1] * a[1][2] * a[2][0]
    d3 = a[0][2] * a[1][0] * a[2][1]
    # Secundarias: a02·a11·a20, a00·a12·a21, a01·a10·a22
    e1 = a[0][2] * a[1][1] * a[2][0]
    e2 = a[0][0] * a[1][2] * a[2][1]
    e3 = a[0][1] * a[1][0] * a[2][2]

    pasos.append("\nDiagonales  (principales):")
    pasos.append(f"  {num(a[0][0])}·{num(a[1][1])}·{num(a[2][2])} = {num(d1)}")
    pasos.append(f"  {num(a[0][1])}·{num(a[1][2])}·{num(a[2][0])} = {num(d2)}")
    pasos.append(f"  {num(a[0][2])}·{num(a[1][0])}·{num(a[2][1])} = {num(d3)}")
    pos = d1 + d2 + d3
    pasos.append(f"  Suma = {num(d1)} + {num(d2)} + {num(d3)} = {num(pos)}")

    pasos.append("\nDiagonales  (secundarias):")
    pasos.append(f"  {num(a[0][2])}·{num(a[1][1])}·{num(a[2][0])} = {num(e1)}")
    pasos.append(f"  {num(a[0][0])}·{num(a[1][2])}·{num(a[2][1])} = {num(e2)}")
    pasos.append(f"  {num(a[0][1])}·{num(a[1][0])}·{num(a[2][2])} = {num(e3)}")
    neg = e1 + e2 + e3
    pasos.append(f"  Suma = {num(e1)} + {num(e2)} + {num(e3)} = {num(neg)}")

    # 4) Desarrollo final estilo pizarra
    pasos.append("\nDesarrollo:")
    pasos.append(f"  |A| = (Suma ) − (Suma )")
    pasos.append(f"= ({num(d1)} + {num(d2)} + {num(d3)}) − ({num(e1)} + {num(e2)} + {num(e3)})")
    pasos.append(f"= {num(pos)} − {num(neg)}")

    val = pos - neg
    pasos.append(f"= {num(val)}")

    return {
        "metodo": "sarrus",
        "det": val,
        "pasos": pasos,
        "conclusion": f"|A| = {fmt_number(val, dec)}"
    }

# ---------- Helper: determinante 3x3 por Sarrus (vertical) con pasos ----------
def _det3_sarrus_vertical(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Calcula el determinante 3x3 por Sarrus (vertical) y genera 
    pasos detallados, imitando el formato de las imágenes.
    Ej: (4 - 18 + 1) - (4 + 6 - 3)
    """
    
    # Validar que sea 3x3 (aunque la función padre ya lo hizo)
    if len(A) != 3 or len(A[0]) != 3:
        raise ValueError("Esta función Sarrus es solo para 3x3.")

    # Extraer los 9 números
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]

    # 1. Calcular productos de diagonales principales (positivas)
    # (a*e*i), (d*h*c), (g*b*f)
    p1 = a * e * i
    p2 = d * h * c
    p3 = g * b * f
    
    # 2. Calcular productos de diagonales secundarias (negativas)
    # (c*e*g), (f*h*a), (i*b*d)
    s1 = c * e * g
    s2 = f * h * a
    s3 = i * b * d

    # 3. Sumar
    sum_pos = p1 + p2 + p3
    sum_neg = s1 + s2 + s3
    
    # 4. Total
    det = sum_pos - sum_neg

    # 5. Generar los pasos (el formato de las imágenes)
    pasos = []
    
    # Formatear los productos individuales
    fp1, fp2, fp3 = fmt_number(p1, dec), fmt_number(p2, dec), fmt_number(p3, dec)
    fs1, fs2, fs3 = fmt_number(s1, dec), fmt_number(s2, dec), fmt_number(s3, dec)

    # Paso 1: Mostrar la línea de cálculo principal
    # Usamos paréntesis para números negativos para mayor claridad
    str_pos_nums = f"{fp1} + ({fp2}) + {fp3}" if p2 < 0 else f"{fp1} + {fp2} + {fp3}"
    str_pos_nums = str_pos_nums.replace(" + -", " - ") # Limpieza simple
    
    str_neg_nums = f"{fs1} + {fs2} + ({fs3})" if s3 < 0 else f"{fs1} + {fs2} + {fs3}"
    str_neg_nums = str_neg_nums.replace(" + -", " - ") # Limpieza simple

    pasos.append(f"   Cálculo: (Diagonales +) - (Diagonales -)")
    pasos.append(f"   = [{str_pos_nums}] - [{str_neg_nums}]")

    # Paso 2: Mostrar los resultados intermedios
    pasos.append(f"   = {fmt_number(sum_pos, dec)} - ({fmt_number(sum_neg, dec)})")
    
    # Paso 3: Mostrar el resultado final
    pasos.append(f"   = {fmt_number(det, dec)}")
    
    return {"det": det, "pasos": pasos}
# ---------- Helper: determinante 2x2 por fórmula con pasos ----------
def _det2_formula(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Calcula el determinante 2x2 por ad-bc y genera pasos.
    """
    if len(A) != 2 or len(A[0]) != 2:
        raise ValueError("Esta función es solo para 2x2.")
    
    # Extraer los 4 números
    a, b = A[0]
    c, d = A[1]
    
    det = (a * d) - (b * c)
    
    # Formatear
    fa, fb = fmt_number(a, dec), fmt_number(b, dec)
    fc, fd = fmt_number(c, dec), fmt_number(d, dec)
    f_prod1 = fmt_number(a * d, dec)
    f_prod2 = fmt_number(b * c, dec)

    pasos = []
    pasos.append(f"   Cálculo: (a·d) - (b·c)")
    pasos.append(f"   = ({fa} · {fd}) - ({fb} · {fc})")
    
    # Añade paréntesis al segundo producto si es negativo
    if (b * c) < 0:
        pasos.append(f"   = {f_prod1} - ({f_prod2})")
    else:
        pasos.append(f"   = {f_prod1} - {f_prod2}")
        
    pasos.append(f"   = {fmt_number(det, dec)}")
    
    return {"det": det, "pasos": pasos}

# ---------- Cramer (ilustrativo): det(A) y det(A_j(b)) con pasos detallados ----------
# ---------- Cramer (ilustrativo): det(A) y det(A_j(b)) con pasos detallados ----------
def det_por_cramer(A: Matriz, b: List[float] | None = None, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Implementación ilustrativa de Cramer para 2x2 y 3×3.
    MODIFICADA: Esta versión solo calcula y muestra |A| por Sarrus (3x3) o ad-bc (2x2).
    """
    # Validaciones básicas
    if not A or not A[0]:
        raise ValueError("Matriz A vacía.")
    n = len(A)
    if any(len(f) != len(A[0]) for f in A) or len(A) != len(A[0]):
        raise ValueError("A debe ser cuadrada.")

    # --- MODIFICACIÓN CLAVE 1: Aceptar 2x2 o 3x3 ---
    if n not in (2, 3):
        # Calculamos el det real por cofactores para devolverlo y evitar el KeyError
        detA_err = det_cofactores(A, dec=dec)["det"] 
        return {
            "metodo": "cramer",
            "det": detA_err, # <-- SE AÑADE ESTA LÍNEA PARA EVITAR EL CRASH
            "pasos": [f"Vista ilustrativa por diagonales disponible solo para 2×2 o 3×3 (n={n}).",
                      "El valor del determinante mostrado se calculó por 'Cofactores'."],
            "conclusion": "Use 'Cofactores' para esta demostración."
        }
    # --- FIN DE LA MODIFICACIÓN 1 ---

    pasos: List[str] = ["--- Regla de Cramer (determinantes por diagonales) ---"]
    pasos.append("A (matriz de coeficientes):\n" + format_matrix_bracket(A, dec))

    # --- MODIFICACIÓN 2: Seleccionar helper 2x2 o 3x3 ---
    if n == 2:
        pasos.append("\nCálculo de |A| por fórmula (ad-bc):")
        detA_info = _det2_formula(A, dec=dec) # <-- LLAMAR A LA NUEVA FUNCIÓN 2x2
    else:  # n == 3
        pasos.append("\nCálculo de |A| por diagonales (Sarrus):")
        detA_info = _det3_sarrus_vertical(A, dec=dec) # <-- LLAMAR A LA FUNCIÓN EXISTENTE
    
    detA = detA_info["det"]
    pasos += detA_info["pasos"]
    pasos.append(f"\nConclusión: |A| = {fmt_number(detA, dec)}")
    # --- FIN DE LA MODIFICACIÓN 2 ---

    out: Dict[str, Any] = {"metodo": "cramer", "det": detA, "pasos": pasos, "conclusion": f"|A| = {fmt_number(detA, dec)}"}

    # --- MODIFICACIÓN CLAVE (existente) ---
    # Terminamos aquí para mostrar solo el cálculo de |A|
    return out
    # --- FIN DE LA MODIFICACIÓN ---

    # El siguiente código (cálculo de A_j y x_j) ya no se ejecutará.
    if b is None:
        return out
    if len(b) != n:
        raise ValueError("Dimensión de b incompatible con A.")

    # ... (todo el resto de la función original) ...

# ---------- Interpretación de invertibilidad ----------
def interpretar_invertibilidad(detA: float, dec: int = DEFAULT_DEC) -> str:
    if abs(detA) < 1e-12:
        return "det|A| = 0 → A es singular, NO tiene inversa."
    return "det|A| ≠ 0 → A es no singular (invertible)."

# ---------- Propiedades del determinante ----------
def propiedad_fila_col_cero(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    _check_square(A)
    n = len(A)
    pasos: List[str] = ["Matriz A:\n" + format_matrix_bracket(A, dec)]
    fila_cero = next((i for i in range(n) if all(abs(x) < 1e-15 for x in A[i])), None)
    col_cero = None
    if fila_cero is None:
        for j in range(n):
            if all(abs(A[i][j]) < 1e-15 for i in range(n)):
                col_cero = j; break
    if fila_cero is not None:
        pasos.append(f"Fila {fila_cero+1} es cero ⇒ det|A|=0.")
    elif col_cero is not None:
        pasos.append(f"Columna {col_cero+1} es cero ⇒ det|A|=0.")
    else:
        pasos.append("No hay fila/columna cero; se calcula det|A| para verificar.")
    detA = det_cofactores(A, dec=dec)["det"]
    conclusion = "Se cumple: fila/columna cero ⇒ det|A|=0." if (fila_cero is not None or col_cero is not None) else \
                 "No hay fila/columna cero; la propiedad aplica cuando exista tal fila/columna."
    return {"propiedad": 1, "det": detA, "pasos": pasos,
            "conclusion": f"{conclusion}  det|A|={fmt_number(detA,dec)}"}

def propiedad_filas_prop(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    _check_square(A)
    n = len(A)
    pasos: List[str] = ["Matriz A:\n" + format_matrix_bracket(A, dec)]
    def _proporcionales(v,w):
        k = None
        for a,b in zip(v,w):
            if abs(b) < 1e-15 and abs(a) < 1e-15:
                continue
            if abs(b) < 1e-15:
                return False
            r = a/b
            if k is None: k = r
            else:
                if abs(r-k) > 1e-12: return False
        return True if k is not None or all(abs(x)<1e-15 for x in v+w) else False
    has = False
    for i in range(n):
        for j in range(i+1,n):
            if _proporcionales(A[i],A[j]):
                pasos.append(f"Filas {i+1} y {j+1} son iguales/proporcionales ⇒ det|A|=0.")
                has = True
    if not has:
        pasos.append("No se detectaron filas proporcionales en la matriz dada.")
    detA = det_cofactores(A, dec=dec)["det"]
    conclusion = "Se cumple: hay filas/columnas proporcionales ⇒ det|A|=0." if has else \
                 "No hubo filas proporcionales; la propiedad aplica cuando existan."
    return {"propiedad": 2, "det": detA, "pasos": pasos,
            "conclusion": f"{conclusion}  det|A|={fmt_number(detA,dec)}"}

def propiedad_swap_signo(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    _check_square(A)
    pasos: List[str] = ["Matriz A:\n" + format_matrix_bracket(A, dec)]
    B = [fila[:] for fila in A]
    if len(B) >= 2:
        B[0], B[1] = B[1], B[0]
    pasos.append("Intercambio de filas F1 ↔ F2 → matriz B:\n" + format_matrix_bracket(B, dec))
    detA = det_cofactores(A, dec=dec)["det"]
    detB = det_cofactores(B, dec=dec)["det"]
    pasos.append(f"det|A|={fmt_number(detA,dec)}; det|B|={fmt_number(detB,dec)}")
    ok = abs(detB + detA) < 1e-9
    concl = "Se cumple: det|B| = -det|A|." if ok else "No se verificó el cambio de signo (revisa datos numéricos)."
    return {"propiedad": 3, "pasos": pasos, "|A|": detA, "|B|": detB, "conclusion": concl}

def propiedad_multiplicar_fila(A: Matriz, k: float, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    _check_square(A)
    pasos: List[str] = ["Matriz A:\n" + format_matrix_bracket(A, dec)]
    if len(A) == 0:
        raise ValueError("Matriz vacía.")
    B = [fila[:] for fila in A]
    for j in range(len(B[0])): B[0][j] *= k
    pasos.append(f"Multiplicar F1 por k={fmt_number(k,dec)} → matriz B:\n" + format_matrix_bracket(B, dec))
    detA = det_cofactores(A, dec=dec)["det"]
    detB = det_cofactores(B, dec=dec)["det"]
    pasos.append(f"det|A|={fmt_number(detA,dec)}; det|B|={fmt_number(detB,dec)}")
    ok = abs(detB - k*detA) < 1e-9
    concl = "Se cumple: det|B| = k·det|A|." if ok else "No se verificó proporcionalidad exacta (numérico)."
    return {"propiedad": 4, "pasos": pasos, "|A|": detA, "|B|": detB, "conclusion": concl}

def propiedad_multiplicativa(A: Matriz, B: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    _check_square(A); _check_square(B)
    if len(A) != len(B):
        raise ValueError("A y B deben ser cuadradas del mismo orden.")
    n = len(A)
    pasos: List[str] = [
        "Matriz A:\n" + format_matrix_bracket(A, dec),
        "\nMatriz B:\n" + format_matrix_bracket(B, dec)
    ]
    # producto clásico (sin numpy)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]
    pasos.append("\nProducto C = AB:\n" + format_matrix_bracket(C, dec))

    detA = det_cofactores(A, dec=dec)["det"]
    detB = det_cofactores(B, dec=dec)["det"]
    detC = det_cofactores(C, dec=dec)["det"]
    pasos.append(f"\ndet|A|={fmt_number(detA,dec)}, det|B|={fmt_number(detB,dec)}, det|AB|={fmt_number(detC,dec)}")
    ok = abs(detC - detA*detB) < 1e-8
    concl = "Se cumple: det|AB| = det|A|·det|B|." if ok else "No se verificó igualdad exacta (afecta redondeo)."
    return {"propiedad": 5, "pasos": pasos, "|A|": detA, "|B|": detB, "|AB|": detC, "conclusion": concl}
