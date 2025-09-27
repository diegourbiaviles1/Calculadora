# algebra_vector.py
from typing import List, Dict, Any
from sistema_lineal import SistemaLineal
from utilidad import (
    is_close, vec_suma, escalar_por_vector, zeros,
    sumar_vec, dot_mat_vec, columnas, format_col_vector, format_matrix_bracket
)

# 1) Propiedades en R^n
def verificar_propiedades(v: List[float], u: List[float], w: List[float], a: float, b: float, tol: float = 1e-10) -> Dict[str, bool]:
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


# 2) Combinación lineal c1 v1 + ... + ck vk
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
    """
    Explica el procedimiento de la combinación lineal y muestra las operaciones de filas
    y el resultado tanto en detalle como en formato simple.
    """
    if len(vectores) != len(coef):
        raise ValueError("Tamaños incompatibles entre vectores y coeficientes.")
    if not vectores:
        return {"resultado": [], "texto": "No se proporcionaron vectores."}

    n = len(vectores[0])  # Dimensión de los vectores
    for v in vectores:
        if len(v) != n:
            raise ValueError("Vectores de distinta dimensión.")

    k = len(vectores)  # Cantidad de vectores
    res = [0] * n  # Vector de resultados

    # Crear el texto para el procedimiento
    txt = []
    txt.append("--- Combinación lineal ---")
    txt.append(f"Cantidad de vectores k: {k}")
    txt.append(f"Dimensión n: {n}\n")

    # Mostrar vectores v1, v2, ..., vk
    for j, vj in enumerate(vectores, start=1):
        txt.append(f"v{j} =\n" + format_col_vector(vj, dec))

    # Mostrar coeficientes
    txt.append("\nCoeficientes:")
    txt.append("c = [ " + "  ".join(str(c) for c in coef) + " ]^T")

    # Fórmula simbólica
    partes = [f"c{j}·v{j}" for j in range(1, k+1)]
    txt.append("\nFórmula de la combinación:")
    txt.append("b = " + " + ".join(partes))

    # Cálculo componente a componente con operaciones
    txt.append("\nCálculo componente a componente:")
    for i in range(n):
        sumandos = [f"{coef[j]}·{vectores[j][i]}" for j in range(k)]
        res[i] = sum([coef[j] * vectores[j][i] for j in range(k)])  # Realizamos el cálculo correctamente
        txt.append(f"b{i+1} = " + " + ".join(sumandos) + f" = {res[i]}")

    # Resultado
    txt.append("\nResultado:")
    txt.append(format_col_vector(res, dec))

    # Resultado simple (sin pasos detallados)
    resultado_simple = f"Como lista: {res}"

    return {"resultado": res, "texto": "\n".join(txt), "resultado_simple": resultado_simple}





# 3) ¿b está en span{v1,...,vk}? -> A c = b con A=[v1 ... vk]
def ecuacion_vectorial(vectores: List[List[float]], b: List[float]) -> Dict[str, Any]:
    if not vectores:
        return {"estado": "sin_vectores"}
    n = len(vectores[0])
    for v in vectores:
        if len(v) != n:
            raise ValueError("Vectores de distinta dimensión")
    if len(b) != n:
        raise ValueError("b tiene dimensión incompatible")

    # construir matriz aumentada [A | b]
    A = zeros(n, len(vectores))
    for j, v in enumerate(vectores):
        for i in range(n):
            A[i][j] = float(v[i])
    Ab = [fila + [float(b[i])] for i, fila in enumerate(A)]

    sis = SistemaLineal(Ab, decimales=4)
    out = sis.gauss_jordan()
    out["reportes"] = out.get("pasos", [])
    return out


# 4) Resolver AX=B para x (o X si B tiene varias columnas)
def resolver_AX_igual_B(A: List[List[float]], B) -> Dict[str, Any]:
    # B puede ser vector (n) o matriz (n x p) tratada como varias columnas
    if isinstance(B[0], list):
        # matriz de columnas
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
        # empaquetar columnas
        n = len(res_cols[0])
        p = len(res_cols)
        X = zeros(n, p)
        for j in range(p):
            for i in range(n):
                X[i][j] = res_cols[j][i]
        return {"estado": "ok", "X": X, "reportes": reportes}
    else:
        # vector b
        Ab = [fila[:] + [B[i]] for i, fila in enumerate(A)]
        sl = SistemaLineal(Ab, decimales=4)
        out = sl.gauss_jordan()
        if out["tipo"] != "unica":
            return {"estado": out["tipo"], "x": None, "reportes": out.get("pasos", [])}
        return {"estado": "ok", "x": out["x"], "reportes": out.get("pasos", [])}


# 5) Distributiva A(u+v)=Au+Av con procedimiento
def verificar_distributiva_matriz(A: List[List[float]], u: List[float], v: List[float], dec: int = 4) -> Dict[str, Any]:
    """
    Verifica A(u+v)=Au+Av mostrando TODO el procedimiento y una conclusión.
    Devuelve: {'cumple': bool, 'pasos': [str], 'conclusion': str}
    """
    pasos: List[str] = []
    pasos.append("Propiedad a verificar: A(u + v) = Au + Av\n")

    pasos.append("Datos:")
    pasos.append("A =\n" + format_matrix_bracket(A, dec))
    pasos.append("u =\n" + format_col_vector(u, dec))
    pasos.append("v =\n" + format_col_vector(v, dec))

    # 1) Calcular u+v
    u_mas_v = sumar_vec(u, v)
    pasos.append("\n1) Suma de vectores u + v:")
    pasos.append("u + v =\n" + format_col_vector(u_mas_v, dec))

    # 2) Calcular A(u+v)
    A_por_u_mas_v = dot_mat_vec(A, u_mas_v)
    pasos.append("\n2) Producto A(u + v):")
    pasos.append("A(u+v) =\n" + format_col_vector(A_por_u_mas_v, dec))

    # 3) Calcular Au y Av por separado
    Au = dot_mat_vec(A, u)
    Av = dot_mat_vec(A, v)
    pasos.append("\n3) Productos por separado:")
    pasos.append("Au =\n" + format_col_vector(Au, dec))
    pasos.append("Av =\n" + format_col_vector(Av, dec))

    # 4) Sumar Au + Av
    Au_mas_Av = sumar_vec(Au, Av)
    pasos.append("\n4) Suma Au + Av:")
    pasos.append("Au + Av =\n" + format_col_vector(Au_mas_Av, dec))

    # 5) Comparación y conclusión
    cumple = all(abs(A_por_u_mas_v[i] - Au_mas_Av[i]) < 1e-10 for i in range(len(A_por_u_mas_v)))
    conclusion = (
        "Se cumple la propiedad distributiva A(u+v) = Au + Av."
        if cumple else
        "No se cumple la propiedad distributiva para los datos dados."
    )
    pasos.append("\n5) Comparación:")
    pasos.append("¿A(u+v) y (Au+Av) son iguales componente a componente?: " + ("Sí" if cumple else "No"))
    pasos.append("\nConclusión: " + conclusion)

    return {"cumple": cumple, "pasos": pasos, "conclusion": conclusion}


# 6) Sistema → forma matricial Ax = b (identificar A, x, b)
def sistema_a_forma_matricial(coefs: List[List[float]], terminos: List[float], nombres_vars: List[str]) -> Dict[str, Any]:
    """
    coefs: matriz m×n con coeficientes (cada fila es una ecuación).
    terminos: vector b de tamaño m.
    nombres_vars: lista de nombres de variables en orden, p.ej. ['x','y','z'].
    """
    if len(coefs) == 0 or len(coefs) != len(terminos):
        raise ValueError("Dimensiones inconsistentes entre coeficientes y términos independientes.")
    m, n = len(coefs), len(coefs[0])
    if len(nombres_vars) != n:
        raise ValueError("La cantidad de nombres de variables no coincide con el número de columnas.")

    A = [fila[:] for fila in coefs]
    x_vars = nombres_vars[:]
    b = terminos[:]

    def _format_col_symbols(names: List[str]) -> str:
        # Imprime un vector columna con símbolos (sin convertir a float)
        return "\n".join([f"[{s}]" for s in names])

    # Texto bonito
    txt: List[str] = []
    txt.append("Forma matricial Ax = b")
    txt.append("\nA (matriz de coeficientes) =\n" + format_matrix_bracket(A))
    txt.append("\nx (vector incógnita) =\n" + _format_col_symbols(x_vars))
    txt.append("\nb (términos independientes) =\n" + format_col_vector(b))
    return {"A": A, "x": x_vars, "b": b, "texto": "\n".join(txt)}


# 7) Matriz·vector explicado: resultado + combinación de columnas
def multiplicacion_matriz_vector_explicada(A: List[List[float]], v: List[float], dec: int = 4) -> Dict[str, Any]:
    """
    Devuelve:
      - 'resultado': A·v (vector)
      - 'texto': procedimiento listo para imprimir,
                 incluyendo combinación lineal de columnas y cálculo fila a fila.
    """
    if len(A) == 0:
        raise ValueError("Matriz vacía.")
    m, n = len(A), len(A[0])
    if len(v) != n:
        raise ValueError("Dimensiones incompatibles entre A y v.")

    res = dot_mat_vec(A, v)
    cols = columnas(A)

    # texto
    txt: List[str] = []
    txt.append("Multiplicación A·v")
    txt.append("\nA =\n" + format_matrix_bracket(A, dec))
    txt.append("\nv =\n" + format_col_vector(v, dec))

    # combinación lineal de columnas
    txt.append("\nInterpretación como combinación lineal de columnas:")
    txt.append("Sea A = [ c1  c2  ...  cn ]. Entonces A·v = v1·c1 + v2·c2 + ... + vn·cn.")
    txt.append("En este caso:")
    suma_str = []
    for j, (col, esc) in enumerate(zip(cols, v), start=1):
        suma_str.append(f"{esc}·c{j}")
        txt.append(f"c{j} =\n" + format_col_vector(col, dec))
    txt.append("Por lo tanto: A·v = " + " + ".join(suma_str))

    # cálculo fila a fila
    txt.append("\nCálculo numérico (fila a fila):")
    for i in range(m):
        fila_terms = [f"{A[i][j]}·{v[j]}" for j in range(n)]
        txt.append(f"fila {i+1}: " + " + ".join(fila_terms) + f" = {res[i]}")

    txt.append("\nResultado:")
    txt.append(format_col_vector(res, dec))

    return {"resultado": res, "texto": "\n".join(txt)}
