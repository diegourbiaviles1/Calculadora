# matrices.py
# -*- coding: utf-8 -*-
"""
Operaciones con matrices y verificación de propiedades de la traspuesta.
Sin NumPy. Pensado para integrarse con tu calculadora por menús.
Todas las rutinas devuelven dicts con 'pasos', 'resultado' (si aplica) y 'conclusion'.
"""

from typing import List, Tuple, Dict, Any

Matriz = List[List[float]]
EPS = 1e-9  # tolerancia para comparaciones

# ---------- Utilidades básicas ----------

def es_rectangular(A: Matriz) -> bool:
    return all(len(f) == len(A[0]) for f in A) if A else False

def dims(A: Matriz) -> Tuple[int, int]:
    return (len(A), len(A[0]) if A else 0)

def mismas_dim(A: Matriz, B: Matriz) -> bool:
    return dims(A) == dims(B)

def compatibles_prod(A: Matriz, B: Matriz) -> bool:
    return dims(A)[1] == dims(B)[0]

def copia(A: Matriz) -> Matriz:
    return [fila[:] for fila in A]

def iguales(A: Matriz, B: Matriz, eps: float = EPS) -> bool:
    if not mismas_dim(A, B):
        return False
    m, n = dims(A)
    for i in range(m):
        for j in range(n):
            if abs(A[i][j] - B[i][j]) > eps:
                return False
    return True

# ---------- Operaciones primitivas ----------

def traspuesta(A: Matriz) -> Matriz:
    m, n = dims(A)
    return [[A[i][j] for i in range(m)] for j in range(n)]

def suma(A: Matriz, B: Matriz) -> Matriz:
    if not mismas_dim(A, B):
        raise ValueError("Para A + B se requieren dimensiones iguales.")
    m, n = dims(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]

def resta(A: Matriz, B: Matriz) -> Matriz:
    if not mismas_dim(A, B):
        raise ValueError("Para A - B se requieren dimensiones iguales.")
    m, n = dims(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]

def escalar_por_matriz(k: float, A: Matriz) -> Matriz:
    m, n = dims(A)
    return [[k * A[i][j] for j in range(n)] for i in range(m)]

def producto(A: Matriz, B: Matriz) -> Matriz:
    if not compatibles_prod(A, B):
        raise ValueError("Para AB, #columnas(A) debe igualar #filas(B).")
    m, p = dims(A)
    p2, n = dims(B)
    assert p == p2
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for k in range(p):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]
    return C

# ---------- Wrappers "explicados" (con pasos y conclusiones) ----------

def suma_matrices_explicada(A: Matriz, B: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not mismas_dim(A, B):
        return {"pasos": pasos, "conclusion": "No se pueden sumar: dimensiones distintas."}
    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para suma.")
    C = suma(A, B)
    pasos.append("C = A + B calculada elemento a elemento.")
    # Propiedad de la traspuesta en la suma
    CT = traspuesta(C)
    AT, BT = traspuesta(A), traspuesta(B)
    suma_Ts = suma(AT, BT)
    cumple = iguales(CT, suma_Ts)
    pasos.append("Se calculó (A + B)^T y A^T + B^T para comparar.")
    conclusion = "Se cumple la propiedad (A + B)^T = A^T + B^T." if cumple else \
                 "No se cumple la propiedad (A + B)^T = A^T + B^T."
    return {
        "pasos": pasos,
        "resultado": C,
        "traspuesta_del_resultado": CT,
        "AT": AT, "BT": BT, "AT_mas_BT": suma_Ts,
        "conclusion": conclusion
    }

def resta_matrices_explicada(A: Matriz, B: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not mismas_dim(A, B):
        return {"pasos": pasos, "conclusion": "No se pueden restar: dimensiones distintas."}
    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para resta.")
    C = resta(A, B)
    pasos.append("C = A - B calculada elemento a elemento.")
    # Propiedad: (A - B)^T = A^T - B^T
    CT = traspuesta(C)
    AT, BT = traspuesta(A), traspuesta(B)
    resta_Ts = resta(AT, BT)
    cumple = iguales(CT, resta_Ts)
    pasos.append("Se calculó (A - B)^T y A^T - B^T para comparar.")
    conclusion = "Se cumple la propiedad (A - B)^T = A^T - B^T." if cumple else \
                 "No se cumple la propiedad (A - B)^T = A^T - B^T."
    return {
        "pasos": pasos,
        "resultado": C,
        "traspuesta_del_resultado": CT,
        "AT": AT, "BT": BT, "AT_menos_BT": resta_Ts,
        "conclusion": conclusion
    }

def producto_escalar_explicado(k: float, A: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A):
        return {"pasos": pasos, "conclusion": "La matriz A no es rectangular."}
    pasos.append(f"Se multiplicó cada entrada de A por k = {k:g}.")
    C = escalar_por_matriz(k, A)
    # Propiedad: (kA)^T = kA^T
    CT = traspuesta(C)
    AT = traspuesta(A)
    kAT = escalar_por_matriz(k, AT)
    cumple = iguales(CT, kAT)
    pasos.append("Se comparó (kA)^T con k·A^T.")
    conclusion = "Se cumple la propiedad (kA)^T = kA^T." if cumple else \
                 "No se cumple la propiedad (kA)^T = kA^T."
    return {
        "pasos": pasos,
        "resultado": C,
        "traspuesta_del_resultado": CT,
        "AT": AT, "kAT": kAT,
        "conclusion": conclusion
    }

def producto_matrices_explicado(A: Matriz, B: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not compatibles_prod(A, B):
        return {"pasos": pasos, "conclusion": "No se puede multiplicar: columnas(A) ≠ filas(B)."}
    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para producto.")
    C = producto(A, B)
    pasos.append("C = A·B con regla i,j: C[i][j] = sum_k A[i][k]·B[k][j].")
    # Propiedad: (AB)^T = B^T A^T
    CT = traspuesta(C)
    AT, BT = traspuesta(A), traspuesta(B)
    BT_AT = producto(BT, AT)
    cumple = iguales(CT, BT_AT)
    pasos.append("Se calculó (AB)^T y B^T·A^T para comparar.")
    conclusion = "Se cumple la propiedad (AB)^T = B^T·A^T." if cumple else \
                 "No se cumple la propiedad (AB)^T = B^T·A^T."
    return {
        "pasos": pasos,
        "resultado": C,
        "traspuesta_del_resultado": CT,
        "AT": AT, "BT": BT, "BT_por_AT": BT_AT,
        "conclusion": conclusion
    }

def traspuesta_explicada(A: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A):
        return {"pasos": pasos, "conclusion": "La matriz A no es rectangular."}
    m, n = dims(A)
    pasos.append(f"Se intercambian filas↔columnas: A es {m}×{n}, A^T será {n}×{m}.")
    AT = traspuesta(A)
    # Propiedad: (A^T)^T = A
    ATT = traspuesta(AT)
    cumple = iguales(ATT, A)
    pasos.append("Verificación: (A^T)^T comparada con A.")
    conclusion = "Se cumple la propiedad (A^T)^T = A." if cumple else \
                 "No se cumple la propiedad (A^T)^T = A."
    return {
        "pasos": pasos,
        "resultado": AT,
        "ATT": ATT,
        "conclusion": conclusion
    }

def verificar_propiedades_traspuesta(A: Matriz, B: Matriz, k: float) -> Dict[str, Any]:
    """
    Consolidado a), b), (versión resta), c), d).
    """
    info: Dict[str, Any] = {}

    # a) (A^T)^T = A
    ATT = traspuesta(traspuesta(A))
    cumple_a = iguales(ATT, A)
    info["a"] = {"se_cumple": cumple_a,
                 "mensaje": "Se cumple la propiedad (A^T)^T = A." if cumple_a
                            else "No se cumple la propiedad (A^T)^T = A."}

    # b) (A + B)^T = A^T + B^T (si aplica)
    if mismas_dim(A, B):
        izq_b = traspuesta(suma(A, B))
        der_b = suma(traspuesta(A), traspuesta(B))
        cumple_b = iguales(izq_b, der_b)
        info["b_suma"] = {"se_cumple": cumple_b,
                          "mensaje": "Se cumple la propiedad (A + B)^T = A^T + B^T." if cumple_b
                                     else "No se cumple la propiedad (A + B)^T = A^T + B^T."}
        # extra: resta
        izq_br = traspuesta(resta(A, B))
        der_br = resta(traspuesta(A), traspuesta(B))
        cumple_br = iguales(izq_br, der_br)
        info["b_resta"] = {"se_cumple": cumple_br,
                           "mensaje": "Se cumple la propiedad (A - B)^T = A^T - B^T." if cumple_br
                                      else "No se cumple la propiedad (A - B)^T = A^T - B^T."}
    else:
        info["b_suma"] = {"se_cumple": False, "mensaje": "No aplica: A y B no tienen las mismas dimensiones."}
        info["b_resta"] = {"se_cumple": False, "mensaje": "No aplica: A y B no tienen las mismas dimensiones."}

    # c) (kA)^T = kA^T
    izq_c = traspuesta(escalar_por_matriz(k, A))
    der_c = escalar_por_matriz(k, traspuesta(A))
    cumple_c = iguales(izq_c, der_c)
    info["c"] = {"se_cumple": cumple_c,
                 "mensaje": "Se cumple la propiedad (kA)^T = kA^T." if cumple_c
                            else "No se cumple la propiedad (kA)^T = kA^T."}

    # d) (AB)^T = B^T A^T (si aplica)
    if compatibles_prod(A, B):
        izq_d = traspuesta(producto(A, B))
        der_d = producto(traspuesta(B), traspuesta(A))
        cumple_d = iguales(izq_d, der_d)
        info["d"] = {"se_cumple": cumple_d,
                     "mensaje": "Se cumple la propiedad (AB)^T = B^T·A^T." if cumple_d
                                else "No se cumple la propiedad (AB)^T = B^T·A^T."}
    else:
        info["d"] = {"se_cumple": False, "mensaje": "No aplica: columnas(A) ≠ filas(B)."}

    return info

def propiedad_r_suma_traspuesta_explicada(A: Matriz, B: Matriz, r: float) -> Dict[str, Any]:
    """
    Verifica la propiedad: ( r (A + B) )^T = r ( A^T + B^T )
    Muestra el procedimiento con pasos e intermedios.
    """
    pasos: List[str] = []

    # Validaciones
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not mismas_dim(A, B):
        return {"pasos": pasos, "conclusion": "No se puede aplicar: A y B deben tener las mismas dimensiones."}

    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para suma.")
    pasos.append(f"Escalar r = {r:g}")

    # Lado izquierdo: ( r (A + B) )^T
    S = suma(A, B)
    pasos.append("Se calcula S = A + B.")
    rS = escalar_por_matriz(r, S)
    pasos.append("Se calcula rS = r·(A + B).")
    izq = traspuesta(rS)
    pasos.append("Se calcula izquierda: ( r (A + B) )^T.")

    # Lado derecho: r ( A^T + B^T )
    AT = traspuesta(A)
    BT = traspuesta(B)
    pasos.append("Se calculan A^T y B^T.")
    ATmBT = suma(AT, BT)
    pasos.append("Se calcula (A^T + B^T).")
    der = escalar_por_matriz(r, ATmBT)
    pasos.append("Se calcula derecha: r·(A^T + B^T).")

    # Comparación
    cumple = iguales(izq, der)
    pasos.append("Se comparan ambas matrices resultado.")
    conclusion = "Se cumple la propiedad ( r (A + B) )^T = r ( A^T + B^T )." if cumple else \
                 "No se cumple la propiedad ( r (A + B) )^T = r ( A^T + B^T )."

    return {
        "pasos": pasos,
        "S": S,
        "rS": rS,
        "izquierda": izq,
        "AT": AT,
        "BT": BT,
        "AT_mas_BT": ATmBT,
        "derecha": der,
        "conclusion": conclusion
    }
