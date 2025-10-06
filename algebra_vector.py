from __future__ import annotations
from typing import List, Dict, Any
import math

from sistema_lineal import SistemaLineal
from utilidad import (
    DEFAULT_EPS, is_close, vec_suma, escalar_por_vector, zeros,
    sumar_vec, dot_mat_vec, columnas, format_col_vector, format_matrix_bracket
)

# 1) Propiedades en R^n
def verificar_propiedades(v: List[float], u: List[float], w: List[float], a: float, b: float, tol: float = DEFAULT_EPS) -> Dict[str, bool]:
    res: Dict[str, bool] = {}
    res["conmutativa"] = all(is_close(x, y, tol) for x, y in zip(vec_suma(v, u), vec_suma(u, v)))

    izq = vec_suma(vec_suma(v, u), w)
    der = vec_suma(v, vec_suma(u, w))
    res["asociativa"] = all(is_close(x, y, tol) for x, y in zip(izq, der))

    cero = [0.0] * len(v)
    res["vector_cero"] = all(is_close(x, y, tol) for x, y in zip(vec_suma(v, cero), v))
    res["opuesto"] = all(is_close(x, 0.0, tol) for x in vec_suma(v, escalar_por_vector(-1.0, v)))

    res["a(u+v)=au+av"] = all(
        is_close(x, y, tol)
        for x, y in zip(
            escalar_por_vector(a, vec_suma(u, v)),
            vec_suma(escalar_por_vector(a, u), escalar_por_vector(a, v)),
        )
    )
    res["(a+b)u=au+bu"] = all(
        is_close(x, y, tol)
        for x, y in zip(
            escalar_por_vector(a + b, u),
            vec_suma(escalar_por_vector(a, u), escalar_por_vector(b, u)),
        )
    )
    res["a(bu)=(ab)u"] = all(
        is_close(x, y, tol)
        for x, y in zip(
            escalar_por_vector(a, escalar_por_vector(b, u)),
            escalar_por_vector(a * b, u),
        )
    )
    return res

# 2) Combinación lineal
def combinacion_lineal(vectores: List[List[float]], coef: List[float]) -> List[float]:
    if len(vectores) != len(coef):
        raise ValueError("Tamaños incompatibles")
    if not vectores:
        return []
    n = len(vectores[0])
    for v in vectores:
        if len(v) != n:
            raise ValueError("Vectores de distinta dimensión")
    res = [0.0] * n
    for v, c in zip(vectores, coef):
        for i in range(n):
            res[i] += float(c) * float(v[i])
    return res

def combinacion_lineal_explicada(vectores: List[List[float]], coef: List[float], dec: int = 4) -> Dict[str, Any]:
    if len(vectores) != len(coef):
        raise ValueError("Tamaños incompatibles entre vectores y coeficientes.")
    if not vectores:
        return {"resultado": [], "texto": "No se proporcionaron vectores.", "resultado_simple": "[]"}

    n = len(vectores[0])
    for v in vectores:
        if len(v) != n:
            raise ValueError("Vectores de distinta dimensión.")

    k = len(vectores)
    res = [0.0] * n

    txt = []
    txt.append("--- Combinación lineal ---")
    txt.append(f"Cantidad de vectores k: {k}")
    txt.append(f"Dimensión n: {n}\n")
    for j, vj in enumerate(vectores, start=1):
        txt.append(f"v{j} =\n" + format_col_vector(vj, dec))

    txt.append("\nCoeficientes:")
    txt.append("c = [ " + "  ".join(str(c) for c in coef) + " ]^T")

    partes = [f"c{j}·v{j}" for j in range(1, k+1)]
    txt.append("\nFórmula de la combinación:")
    txt.append("b = " + " + ".join(partes))

    txt.append("\nCálculo componente a componente:")
    for i in range(n):
        sumandos = [f"{coef[j]}·{vectores[j][i]}" for j in range(k)]
        res[i] = math.fsum(coef[j] * vectores[j][i] for j in range(k))
        txt.append(f"b{i+1} = " + " + ".join(sumandos) + f" = {res[i]}")

    txt.append("\nResultado:")
    txt.append(format_col_vector(res, dec))

    return {"resultado": res, "texto": "\n".join(txt), "resultado_simple": str(res)}

# 3) ¿b está en span{v1,...,vk}? -> A c = b
def ecuacion_vectorial(vectores: List[List[float]], b: List[float]) -> Dict[str, Any]:
    if not vectores:
        return {"estado": "sin_vectores"}
    n = len(vectores[0])
    for v in vectores:
        if len(v) != n:
            raise ValueError("Vectores de distinta dimensión")
    if len(b) != n:
        raise ValueError("b tiene dimensión incompatible")

    A = zeros(n, len(vectores))
    for j, v in enumerate(vectores):
        for i in range(n):
            A[i][j] = float(v[i])
    Ab = [fila + [float(b[i])] for i, fila in enumerate(A)]

    sis = SistemaLineal(Ab, decimales=4)
    out = sis.gauss_jordan()
    out["reportes"] = out.get("pasos", [])
    return out

# 4) Resolver AX=B
def resolver_AX_igual_B(A: List[List[float]], B) -> Dict[str, Any]:
    if isinstance(B[0], list):
        res_cols = []
        reportes: List[str] = []
        for col_idx in range(len(B[0])):
            b = [B[i][col_idx] for i in range(len(B))]
            Ab = [fila[:] + [b[i]] for i, fila in enumerate(A)]
            sl = SistemaLineal(Ab, decimales=4)
            out = sl.gauss_jordan()
            reportes += out.get("pasos", [])
            if out["tipo"] == "unica":
                res_cols.append(out["x"])
            else:
                res_cols.append(None)
        if any(c is None for c in res_cols):
            return {"estado": "no_unica_en_alguna_columna", "X": None, "reportes": reportes}
        n = len(res_cols[0]); p = len(res_cols)
        X = zeros(n, p)
        for j in range(p):
            for i in range(n):
                X[i][j] = res_cols[j][i]
        return {"estado": "ok", "X": X, "reportes": reportes}
    else:
        Ab = [fila[:] + [B[i]] for i, fila in enumerate(A)]
        sl = SistemaLineal(Ab, decimales=4)
        out = sl.gauss_jordan()
        if out["tipo"] != "unica":
            return {"estado": out["tipo"], "x": None, "reportes": out.get("pasos", [])}
        return {"estado": "ok", "x": out["x"], "reportes": out.get("pasos", [])}

# 5) Distributiva A(u+v)=Au+Av
def verificar_distributiva_matriz(A: List[List[float]], u: List[float], v: List[float], dec: int = 4) -> Dict[str, Any]:
    pasos: List[str] = []
    pasos.append("Propiedad a verificar: A(u + v) = Au + Av\n")
    pasos.append("Datos:")
    pasos.append("A =\n" + format_matrix_bracket(A, dec))
    pasos.append("u =\n" + format_col_vector(u, dec))
    pasos.append("v =\n" + format_col_vector(v, dec))

    u_mas_v = sumar_vec(u, v)
    pasos.append("\n1) Suma de vectores u + v:")
    pasos.append("u + v =\n" + format_col_vector(u_mas_v, dec))

    A_por_u_mas_v = dot_mat_vec(A, u_mas_v)
    pasos.append("\n2) Producto A(u + v):")
    pasos.append("A(u+v) =\n" + format_col_vector(A_por_u_mas_v, dec))

    Au = dot_mat_vec(A, u)
    Av = dot_mat_vec(A, v)
    pasos.append("\n3) Productos por separado:")
    pasos.append("Au =\n" + format_col_vector(Au, dec))
    pasos.append("Av =\n" + format_col_vector(Av, dec))

    Au_mas_Av = sumar_vec(Au, Av)
    pasos.append("\n4) Suma Au + Av:")
    pasos.append("Au + Av =\n" + format_col_vector(Au_mas_Av, dec))

    cumple = all(is_close(A_por_u_mas_v[i], Au_mas_Av[i], DEFAULT_EPS) for i in range(len(A_por_u_mas_v)))
    conclusion = ("Se cumple la propiedad distributiva A(u+v) = Au + Av."
                  if cumple else "No se cumple la propiedad distributiva para los datos dados.")
    pasos.append("\n5) Comparación:")
    pasos.append("¿A(u+v) y (Au+Av) son iguales componente a componente?: " + ("Sí" if cumple else "No"))
    pasos.append("\nConclusión: " + conclusion)

    return {"cumple": cumple, "pasos": pasos, "conclusion": conclusion}

# 6) Sistema → forma matricial Ax = b
def sistema_a_forma_matricial(coefs: List[List[float]], terminos: List[float], nombres_vars: List[str]) -> Dict[str, Any]:
    if len(coefs) == 0 or len(coefs) != len(terminos):
        raise ValueError("Dimensiones inconsistentes entre coeficientes y términos independientes.")
    m, n = len(coefs), len(coefs[0])
    if len(nombres_vars) != n:
        raise ValueError("La cantidad de nombres de variables no coincide con el número de columnas.")

    A = [fila[:] for fila in coefs]
    x_vars = nombres_vars[:]
    b = terminos[:]

    def _format_col_symbols(names: List[str]) -> str:
        return "\n".join([f"[{s}]" for s in names])

    txt: List[str] = []
    txt.append("Forma matricial Ax = b")
    txt.append("\nA (matriz de coeficientes) =\n" + format_matrix_bracket(A))
    txt.append("\nx (vector incógnita) =\n" + _format_col_symbols(x_vars))
    txt.append("\nb (términos independientes) =\n" + format_col_vector(b))
    return {"A": A, "x": x_vars, "b": b, "texto": "\n".join(txt)}

# 7) Matriz·vector explicado
def multiplicacion_matriz_vector_explicada(A: List[List[float]], v: List[float], dec: int = 4) -> Dict[str, Any]:
    if len(A) == 0:
        raise ValueError("Matriz vacía.")
    m, n = len(A), len(A[0])
    if len(v) != n:
        raise ValueError("Dimensiones incompatibles entre A y v.")

    res = dot_mat_vec(A, v)
    cols = columnas(A)

    txt: List[str] = []
    txt.append("Multiplicación A·v")
    txt.append("\nA =\n" + format_matrix_bracket(A, dec))
    txt.append("\nv =\n" + format_col_vector(v, dec))
    txt.append("\nInterpretación como combinación lineal de columnas:")
    txt.append("Sea A = [ c1  c2  ...  cn ]. Entonces A·v = v1·c1 + v2·c2 + ... + vn·cn.")
    txt.append("En este caso:")
    suma_str = []
    for j, (col, esc) in enumerate(zip(cols, v), start=1):
        suma_str.append(f"{esc}·c{j}")
        txt.append(f"c{j} =\n" + format_col_vector(col, dec))
    txt.append("Por lo tanto: A·v = " + " + ".join(suma_str))
    txt.append("\nCálculo numérico (fila a fila):")
    for i in range(m):
        fila_terms = [f"{A[i][j]}·{v[j]}" for j in range(n)]
        txt.append(f"fila {i+1}: " + " + ".join(fila_terms) + f" = {res[i]}")
    txt.append("\nResultado:")
    txt.append(format_col_vector(res, dec))
    return {"resultado": res, "texto": "\n".join(txt)}
