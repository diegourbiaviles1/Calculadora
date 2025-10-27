# determinante.py — versión con "paso a paso" detallado
from __future__ import annotations
from typing import List, Dict, Any
from utilidad import fmt_number, DEFAULT_DEC, format_matrix_bracket

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

# ---------- Cofactores (Laplace con pasos visibles) ----------
def det_cofactores(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Determinante por expansión de cofactores (Laplace) con TRAZA paso a paso:
      - Muestra la matriz actual
      - Indica la fila elegida para expandir
      - Escribe cada término:  (-1)^{i+j} · a_{ij} · det(M_{ij})
      - Muestra los menores con formato de matriz
      - Casos base (1×1, 2×2) explicados con fórmula explícita
    """
    _check_square(A)
    pasos: List[str] = []

    def _det_rec(M: Matriz, nivel: int) -> float:
        n = len(M)
        indent = "  " * nivel
        pasos.append(indent + "Matriz en este nivel:\n" + indent + format_matrix_bracket(M, dec))

        if n == 1:
            val = M[0][0]
            pasos.append(indent + f"Caso 1×1: det([[{fmt_number(val,dec)}]]) = {fmt_number(val,dec)}")
            return val

        if n == 2:
            a,b = M[0][0], M[0][1]
            c,d = M[1][0], M[1][1]
            val = a*d - b*c
            pasos.append(indent + "Caso 2×2: det([[a,b],[c,d]]) = ad − bc")
            pasos.append(indent + f"= {fmt_number(a,dec)}·{fmt_number(d,dec)} − {fmt_number(b,dec)}·{fmt_number(c,dec)}")
            pasos.append(indent + f"= {fmt_number(val,dec)}")
            return val

        if n == 3:
            a = M
            t1 = a[0][0]*a[1][1]*a[2][2]
            t2 = a[0][1]*a[1][2]*a[2][0]
            t3 = a[0][2]*a[1][0]*a[2][1]
            u1 = a[0][2]*a[1][1]*a[2][0]
            u2 = a[0][0]*a[1][2]*a[2][1]
            u3 = a[0][1]*a[1][0]*a[2][2]
            pasos.append(indent + "Caso 3×3 (fórmula completa):")
            pasos.append(indent + f"+ {fmt_number(a[0][0],dec)}·{fmt_number(a[1][1],dec)}·{fmt_number(a[2][2],dec)}"
                                   f" + {fmt_number(a[0][1],dec)}·{fmt_number(a[1][2],dec)}·{fmt_number(a[2][0],dec)}"
                                   f" + {fmt_number(a[0][2],dec)}·{fmt_number(a[1][0],dec)}·{fmt_number(a[2][1],dec)}")
            pasos.append(indent + f"− {fmt_number(a[0][2],dec)}·{fmt_number(a[1][1],dec)}·{fmt_number(a[2][0],dec)}"
                                   f" − {fmt_number(a[0][0],dec)}·{fmt_number(a[1][2],dec)}·{fmt_number(a[2][1],dec)}"
                                   f" − {fmt_number(a[0][1],dec)}·{fmt_number(a[1][0],dec)}·{fmt_number(a[2][2],dec)}")
            val = (t1+t2+t3) - (u1+u2+u3)
            pasos.append(indent + f"= {fmt_number(val,dec)}")
            return val

        fila = _mejor_fila_para_expandir(M)
        pasos.append(indent + f"Expansión por la fila {fila+1} (se eligió por tener más ceros).")
        total = 0.0
        partes = []
        for j, aij in enumerate(M[fila]):
            if abs(aij) < 1e-15:  # término nulo
                continue
            sgn = _cof_sign(fila, j)
            cof = "(-1)^{i+j} = -1" if sgn < 0 else "(-1)^{i+j} = 1"
            Mm = _minor(M, fila, j)
            pasos.append(indent + f"Término (i={fila+1}, j={j+1}):  {cof},  a[{fila+1},{j+1}]={fmt_number(aij,dec)}")
            pasos.append(indent + "Menor M_{ij} (eliminando fila y columna):\n" + indent + format_matrix_bracket(Mm, dec))
            sub = _det_rec(Mm, nivel+1)
            term = sgn * aij * sub
            partes.append(f"{'+' if sgn>0 else '-'} {fmt_number(abs(aij),dec)}·det(M_{fila+1},{j+1}) = {fmt_number(term,dec)}")
            total += term
        pasos.append(indent + "Suma de términos: " + "  ".join(partes))
        pasos.append(indent + f"det = {fmt_number(total,dec)}")
        return total

    val = _det_rec(A, 0)
    return {"metodo": "cofactores", "det": val, "pasos": pasos,
            "conclusion": f"det(A) = {fmt_number(val, dec)}"}

# ---------- Regla de Sarrus (solo 3×3) con diagonales explícitas ----------
def det_sarrus(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    _check_square(A)
    if len(A) != 3:
        raise ValueError("La regla de Sarrus solo aplica a matrices 3×3.")
    a = A
    pasos: List[str] = []
    pasos.append("Matriz A:\n" + format_matrix_bracket(a, dec))
    pasos.append("Extiende las dos primeras columnas a la derecha (solo para visualizar diagonales).")

    # Diagonales principales
    d1 = a[0][0]*a[1][1]*a[2][2]
    d2 = a[0][1]*a[1][2]*a[2][0]
    d3 = a[0][2]*a[1][0]*a[2][1]
    pasos.append(f"Diagonales principales: "
                 f"{fmt_number(a[0][0],dec)}·{fmt_number(a[1][1],dec)}·{fmt_number(a[2][2],dec)} = {fmt_number(d1,dec)}, "
                 f"{fmt_number(a[0][1],dec)}·{fmt_number(a[1][2],dec)}·{fmt_number(a[2][0],dec)} = {fmt_number(d2,dec)}, "
                 f"{fmt_number(a[0][2],dec)}·{fmt_number(a[1][0],dec)}·{fmt_number(a[2][1],dec)} = {fmt_number(d3,dec)}")
    pos = d1 + d2 + d3
    pasos.append(f"Suma de principales: {fmt_number(pos,dec)}")

    # Diagonales secundarias
    e1 = a[0][2]*a[1][1]*a[2][0]
    e2 = a[0][0]*a[1][2]*a[2][1]
    e3 = a[0][1]*a[1][0]*a[2][2]
    pasos.append(f"Diagonales secundarias: "
                 f"{fmt_number(a[0][2],dec)}·{fmt_number(a[1][1],dec)}·{fmt_number(a[2][0],dec)} = {fmt_number(e1,dec)}, "
                 f"{fmt_number(a[0][0],dec)}·{fmt_number(a[1][2],dec)}·{fmt_number(a[2][1],dec)} = {fmt_number(e2,dec)}, "
                 f"{fmt_number(a[0][1],dec)}·{fmt_number(a[1][0],dec)}·{fmt_number(a[2][2],dec)} = {fmt_number(e3,dec)}")
    neg = e1 + e2 + e3
    pasos.append(f"Suma de secundarias: {fmt_number(neg,dec)}")

    val = pos - neg
    pasos.append(f"det(A) = (suma principales) − (suma secundarias) = {fmt_number(pos,dec)} − {fmt_number(neg,dec)} = {fmt_number(val,dec)}")
    return {"metodo": "sarrus", "det": val, "pasos": pasos,
            "conclusion": f"det(A) = {fmt_number(val, dec)}"}

# ---------- Cramer (ilustrativo: det(A) y det(A_j(b)) con pasos) ----------
def det_por_cramer(A: Matriz, b: List[float] | None = None, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    """
    Muestra el rol de determinantes en Cramer:
      - Calcula det(A) (con pasos de cofactores).
      - Si se da b, construye A_j(b), imprime A_j y calcula det(A_j) con sus propios pasos.
      - Concluye con la fórmula x_j = det(A_j)/det(A) cuando det(A)≠0.
    """
    _check_square(A)
    n = len(A)
    pasos: List[str] = ["--- Método de Cramer (traza de determinantes) ---",
                        "A (matriz de coeficientes):\n" + format_matrix_bracket(A, dec)]
    base = det_cofactores(A, dec=dec)
    detA = base["det"]
    pasos += ["\nCálculo de det(A):"] + base["pasos"]
    if b is not None:
        if len(b) != n:
            raise ValueError("Dimensión de b incompatible.")
        pasos.append("\nVector independiente b:\n" + "\n".join([f"[{fmt_number(x,dec)}]" for x in b]))
        dets_Aj = []
        for j in range(n):
            Aj = [fila[:] for fila in A]
            for i in range(n):
                Aj[i][j] = b[i]
            pasos.append(f"\nReemplazo de columna {j+1} por b → A_{j+1}(b):\n" + format_matrix_bracket(Aj, dec))
            dj = det_cofactores(Aj, dec=dec)
            dets_Aj.append(dj["det"])
            pasos.append(f"Cálculo de det(A_{j+1}):")
            pasos += dj["pasos"]
            pasos.append(f"Resultado: det(A_{j+1}) = {fmt_number(dj['det'],dec)}")
        pasos.append("\nSi det(A) ≠ 0, entonces para cada j:  x_j = det(A_j)/det(A).")
        return {"metodo": "cramer", "det": detA, "detA_j": dets_Aj, "pasos": pasos,
                "conclusion": f"det(A) = {fmt_number(detA, dec)}"}
    return {"metodo": "cramer", "det": detA, "pasos": pasos,
            "conclusion": f"det(A) = {fmt_number(detA, dec)}"}

# ---------- Interpretación de invertibilidad ----------
def interpretar_invertibilidad(detA: float, dec: int = DEFAULT_DEC) -> str:
    if abs(detA) < 1e-12:
        return "det(A) = 0 → A es singular, NO tiene inversa."
    return "det(A) ≠ 0 → A es no singular (invertible)."

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
        pasos.append(f"Fila {fila_cero+1} es cero ⇒ det(A)=0.")
    elif col_cero is not None:
        pasos.append(f"Columna {col_cero+1} es cero ⇒ det(A)=0.")
    else:
        pasos.append("No hay fila/columna cero; se calcula det(A) para verificar.")
    detA = det_cofactores(A, dec=dec)["det"]
    conclusion = "Se cumple: fila/columna cero ⇒ det(A)=0." if (fila_cero is not None or col_cero is not None) else \
                 "No hay fila/columna cero; la propiedad aplica cuando exista tal fila/columna."
    return {"propiedad": 1, "det": detA, "pasos": pasos,
            "conclusion": f"{conclusion}  det(A)={fmt_number(detA,dec)}"}

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
                pasos.append(f"Filas {i+1} y {j+1} son iguales/proporcionales ⇒ det(A)=0.")
                has = True
    if not has:
        pasos.append("No se detectaron filas proporcionales en la matriz dada.")
    detA = det_cofactores(A, dec=dec)["det"]
    conclusion = "Se cumple: hay filas/columnas proporcionales ⇒ det(A)=0." if has else \
                 "No hubo filas proporcionales; la propiedad aplica cuando existan."
    return {"propiedad": 2, "det": detA, "pasos": pasos,
            "conclusion": f"{conclusion}  det(A)={fmt_number(detA,dec)}"}

def propiedad_swap_signo(A: Matriz, dec: int = DEFAULT_DEC) -> Dict[str, Any]:
    _check_square(A)
    pasos: List[str] = ["Matriz A:\n" + format_matrix_bracket(A, dec)]
    B = [fila[:] for fila in A]
    if len(B) >= 2:
        B[0], B[1] = B[1], B[0]
    pasos.append("Intercambio de filas F1 ↔ F2 → matriz B:\n" + format_matrix_bracket(B, dec))
    detA = det_cofactores(A, dec=dec)["det"]
    detB = det_cofactores(B, dec=dec)["det"]
    pasos.append(f"det(A)={fmt_number(detA,dec)}; det(B)={fmt_number(detB,dec)}")
    ok = abs(detB + detA) < 1e-9
    concl = "Se cumple: det(B) = -det(A)." if ok else "No se verificó el cambio de signo (revisa datos numéricos)."
    return {"propiedad": 3, "pasos": pasos, "detA": detA, "detB": detB, "conclusion": concl}

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
    pasos.append(f"det(A)={fmt_number(detA,dec)}; det(B)={fmt_number(detB,dec)}")
    ok = abs(detB - k*detA) < 1e-9
    concl = "Se cumple: det(B) = k·det(A)." if ok else "No se verificó proporcionalidad exacta (numérico)."
    return {"propiedad": 4, "pasos": pasos, "detA": detA, "detB": detB, "conclusion": concl}

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
    pasos.append(f"\ndet(A)={fmt_number(detA,dec)}, det(B)={fmt_number(detB,dec)}, det(AB)={fmt_number(detC,dec)}")
    ok = abs(detC - detA*detB) < 1e-8
    concl = "Se cumple: det(AB) = det(A)·det(B)." if ok else "No se verificó igualdad exacta (afecta redondeo)."
    return {"propiedad": 5, "pasos": pasos, "detA": detA, "detB": detB, "detAB": detC, "conclusion": concl}
