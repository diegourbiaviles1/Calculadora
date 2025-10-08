# programa4.py — Homogéneos / No homogéneos y Dependencia / Independencia
from __future__ import annotations
from typing import List, Dict, Any
from sistema_lineal import SistemaLineal, formatear_solucion_parametrica
from utilidad import mat_from_columns


# -----------------------------------------------------------
# Función auxiliar: detectar si un vector es nulo (Ax = 0)
# -----------------------------------------------------------
def _es_vector_cero(b: List[float], eps: float = 1e-10) -> bool:
    return all(abs(x) < eps for x in b)


# -----------------------------------------------------------
# Analiza sistema Ax = b (sirve para homogéneo o no homogéneo)
# -----------------------------------------------------------
def analizar_sistema(A: List[List[float]], b: List[float]) -> Dict[str, Any]:
    """
    Analiza un sistema lineal Ax=b, usando el método Gauss–Jordan.
    Detecta si es homogéneo, consistente, único o con infinitas soluciones.
    """
    Ab = [fila[:] + [b[i]] for i, fila in enumerate(A)]
    sl = SistemaLineal(Ab, decimales=4)
    out = sl.gauss_jordan()

    homogeneo = _es_vector_cero(b)
    estado = out["tipo"]
    consistente = (estado != "inconsistente")

    if homogeneo:
        solo_trivial = (estado == "unica")
        hay_no_triviales = (estado == "infinitas")
    else:
        solo_trivial = None
        hay_no_triviales = None

    param = formatear_solucion_parametrica(out, nombres_vars=None, dec=4, fracciones=True)

    # --- Conclusión del sistema ---
    if not consistente:
        conclusion = "El sistema es inconsistente: no tiene soluciones."
    elif estado == "unica":
        conclusion = "El sistema es consistente con solución única."
    else:
        conclusion = "El sistema es consistente con infinitas soluciones."

    if homogeneo and consistente:
        conclusion += " (Homogéneo: "
        conclusion += "solo solución trivial)." if solo_trivial else "tiene soluciones no triviales)."

    return {
        "homogeneo": homogeneo,
        "consistente": consistente,
        "tipo": estado,
        "solo_trivial": solo_trivial,
        "hay_no_triviales": hay_no_triviales,
        "salida_parametrica": param,
        "pasos": out.get("pasos", []),
        "pivotes": out.get("pivotes", []),
        "libres": out.get("libres"),
        "x_part": out.get("x_part"),
        "base_nulo": out.get("base_nulo"),
        "rref": out.get("rref"),
        "conclusion": conclusion,
    }


# -----------------------------------------------------------
# Dependencia lineal (resuelve A·c = 0)
# -----------------------------------------------------------
def analizar_dependencia(vectores: List[List[float]]) -> Dict[str, Any]:
    """
    Determina si los vectores dados son linealmente dependientes o independientes,
    resolviendo el sistema homogéneo A·c = 0.
    """
    if not vectores:
        raise ValueError("No se proporcionaron vectores.")

    # Matriz A formada con vectores como columnas
    A = mat_from_columns(vectores)
    b = [0.0] * len(A)  # vector nulo
    Ab = [A[i][:] + [b[i]] for i in range(len(A))]

    sl = SistemaLineal(Ab, decimales=4)
    out = sl.gauss_jordan()

    # Clasificación del tipo de sistema homogéneo
    if out["tipo"] == "unica":
        veredicto = "independientes"
        mensaje = "Los vectores son linealmente independientes (solo solución trivial en A·c = 0)."
    elif out["tipo"] == "infinitas":
        veredicto = "dependientes"
        mensaje = "Los vectores son linealmente dependientes (existen soluciones no triviales en A·c = 0)."
    else:
        veredicto = "indeterminado"
        mensaje = "Resultado inusual: el sistema homogéneo resultó inconsistente."

    param = formatear_solucion_parametrica(
        out, nombres_vars=[f"c{j+1}" for j in range(len(vectores))], dec=4, fracciones=True
    )

    return {
        "veredicto": veredicto,
        "mensaje": mensaje,
        "salida_parametrica": param,
        "pasos": out.get("pasos", []),
        "pivotes": out.get("pivotes", []),
        "libres": out.get("libres"),
        "x_part": out.get("x_part"),
        "base_nulo": out.get("base_nulo"),
        "rref": out.get("rref"),
    }


# -----------------------------------------------------------
# Unifica homogéneo / no homogéneo y dependencia lineal
# -----------------------------------------------------------
def resolver_sistema_homogeneo_y_no_homogeneo(A, b=None):
    """
    Resuelve el sistema Ax=b o Ax=0 (si b es None o vector nulo),
    para luego poder ser usado también en dependencia lineal.
    """
    from homogeneo import analizar_sistema
    if b is None:
        b = [0.0] * len(A)
    return analizar_sistema(A, b)


def resolver_dependencia_lineal_con_homogeneo(A):
    """
    Usa el sistema homogéneo Ax = 0 para mostrar pasos completos
    y determinar si las columnas de A son LI o LD.
    """
    resultado = resolver_sistema_homogeneo_y_no_homogeneo(A)
    if "infinitas" in resultado["conclusion"].lower():
        resultado["dependencia"] = "Conclusión: Vectores columna de A son DEPENDIENTES."
    else:
        resultado["dependencia"] = "Conclusión: Vectores columna de A son INDEPENDIENTES."
    return resultado
