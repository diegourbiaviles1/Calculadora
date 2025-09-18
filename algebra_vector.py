# algebra.py
from sistema_lineal import SistemaLineal
from utilidad import is_close, vec_suma, escalar_por_vector, col_matrix, zeros

# 1) Propiedades en R^n
def verificar_propiedades(v, u, w, a, b, tol=1e-10):
    res = {}
    # Conmutativa
    res["conmutativa"] = all(is_close(x,y,tol) for x,y in zip(vec_suma(v,u), vec_suma(u,v)))
    # Asociativa
    izq = vec_suma(vec_suma(v,u), w)
    der = vec_suma(v, vec_suma(u,w))
    res["asociativa"] = all(is_close(x,y,tol) for x,y in zip(izq,der))
    # Cero
    cero = [0.0]*len(v)
    res["vector_cero"] = all(is_close(x,y,tol) for x,y in zip(vec_suma(v,cero), v))
    # Opuesto
    res["opuesto"] = all(is_close(x,0.0,tol) for x in vec_suma(v, escalar_por_vector(-1.0, v)))
    # Distributivas
    res["a(u+v)=au+av"] = all(is_close(x,y,tol) for x,y in zip(escalar_por_vector(a, vec_suma(u,v)),
                                                               vec_suma(escalar_por_vector(a,u), escalar_por_vector(a,v))))
    res["(a+b)u=au+bu"] = all(is_close(x,y,tol) for x,y in zip(escalar_por_vector(a+b, u),
                                                               vec_suma(escalar_por_vector(a,u), escalar_por_vector(b,u))))
    res["a(bu)=(ab)u"] = all(is_close(x,y,tol) for x,y in zip(escalar_por_vector(a, escalar_por_vector(b,u)),
                                                              escalar_por_vector(a*b, u)))
    return res

# 2) y 3) Combinación lineal / Ecuación vectorial
def combinacion_lineal(vectores, b, decimales=4):
    """
    ¿b está en span{v1,...,vk}? Plantea A c = b con A=[v1...vk] y usa Gauss-Jordan.
    """
    A = col_matrix(vectores)                 # n×k
    Ab = [fila + [b[i]] for i, fila in enumerate(A)]
    sis = SistemaLineal(Ab, decimales=decimales)
    traza = sis.eliminacion_gaussiana()     # incluye interpretación al final

    # Clasificamos por el texto final
    if "inconsistente" in traza:
        estado = "no_pertenece"
    elif "La solución es única" in traza:
        estado = "unica"
    elif "infinitas" in traza:
        estado = "infinitas"
    else:
        estado = "desconocido"

    # Si única, extrae coeficientes
    coef = None
    if estado == "unica":
        R = sis.matriz
        n, m = len(R), len(R[0]) - 1
        coef = [0.0]*m
        for j in range(m):
            i_piv = next((i for i in range(n) if abs(R[i][j]-1.0)<1e-8 and all(abs(R[t][j])<1e-8 for t in range(n) if t!=i)), None)
            if i_piv is not None: coef[j] = R[i_piv][m]

    return {"estado": estado, "coeficientes": coef, "traza": traza}

def ecuacion_vectorial(vectores, b, decimales=4):
    return combinacion_lineal(vectores, b, decimales=decimales)

# 4) Ecuación matricial AX=B
def resolver_AX_igual_B(A, B, decimales=4):
    """
    A es m×n. B puede ser vector (m,) o matriz m×p. Resuelve por columnas.
    """
    m, n = len(A), len(A[0])
    if B and isinstance(B[0], (int, float)):           # vector
        columnas = [[float(x) for x in B]]
    else:                                              # matriz
        p = len(B[0])
        columnas = [[B[i][j] for i in range(m)] for j in range(p)]

    soluciones, reportes = [], []
    for b in columnas:
        Ab = [list(A[i]) + [b[i]] for i in range(m)]
        sis = SistemaLineal(Ab, decimales=decimales)
        traza = sis.eliminacion_gaussiana()
        reportes.append(traza)
        if "inconsistente" in traza or "infinitas" in traza:
            soluciones.append(None)
        else:
            R = sis.matriz
            x = [0.0]*n
            for j in range(n):
                i_piv = next((i for i in range(m) if abs(R[i][j]-1.0)<1e-8 and all(abs(R[t][j])<1e-8 for t in range(m) if t!=i)), None)
                if i_piv is not None: x[j] = R[i_piv][n]
            soluciones.append(x)

    if any(s is None for s in soluciones):
        return {"estado": "no_unica_en_alguna_columna", "X": None, "reportes": reportes}

    p = len(soluciones)
    X = zeros(n, p)
    for j in range(p):
        col = soluciones[j]
        for i in range(n): X[i][j] = col[i]
    return {"estado": "ok", "X": X if p>1 else [row[0] for row in X], "reportes": reportes}
