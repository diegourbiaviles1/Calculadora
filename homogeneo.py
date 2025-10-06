# programa4.py — Homogéneos/No homogéneos y Dependencia/Independencia
from __future__ import annotations
from typing import List, Dict, Any
from sistema_lineal import SistemaLineal, formatear_solucion_parametrica
from utilidad import mat_from_columns

def _es_vector_cero(b: List[float], eps: float = 1e-10) -> bool:
    return all(abs(x) < eps for x in b)

def analizar_sistema(A: List[List[float]], b: List[float]) -> Dict[str, Any]:
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

    if not consistente:
        conclusion = "El sistema es inconsistente: no tiene soluciones."
    elif estado == "unica":
        conclusion = "El sistema es consistente con solución única."
    else:
        conclusion = "El sistema es consistente con infinitas soluciones."

    if homogeneo and consistente:
        conclusion += " (Homogéneo: " + ("solo solución trivial)." if solo_trivial else "tiene soluciones no triviales).")

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

def analizar_dependencia(vectores: List[List[float]]) -> Dict[str, Any]:
    if not vectores:
        raise ValueError("No se proporcionaron vectores.")
    A = mat_from_columns(vectores)          # n x k
    b = [0.0] * len(A)                      # vector cero n
    Ab = [A[i][:] + [b[i]] for i in range(len(A))]

    sl = SistemaLineal(Ab, decimales=4)
    out = sl.gauss_jordan()                 # sobre c (coeficientes)

    if out["tipo"] == "unica":
        veredicto = "independientes"
        mensaje = "Los vectores son linealmente independientes (solo solución trivial en A c = 0)."
    elif out["tipo"] == "infinitas":
        veredicto = "dependientes"
        mensaje = "Los vectores son linealmente dependientes (existen soluciones no triviales en A c = 0)."
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
