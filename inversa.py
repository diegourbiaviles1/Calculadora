
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from utilidad import DEFAULT_EPS, fmt_number, format_matrix_bracket, copy_mat, eye
from sistema_lineal import SistemaLineal, formatear_solucion_parametrica
from homogeneo import analizar_sistema, analizar_dependencia

Matriz = List[List[float]]

# ----------------------------
# Validaciones básicas
# ----------------------------
def _is_rectangular(A: Matriz) -> bool:
    return bool(A) and all(len(f) == len(A[0]) for f in A)

def _check_square(A: Matriz):
    if not _is_rectangular(A):
        raise ValueError("Matriz A inválida o no rectangular.")
    m, n = len(A), len(A[0])
    if m != n:
        raise ValueError("La matriz A debe ser cuadrada para ser invertible.")

# ----------------------------
# Gauss–Jordan sobre [A | I]
# ----------------------------
def inversa_por_gauss_jordan(
    A: Matriz,
    tol: float = DEFAULT_EPS,
    dec: int = 4
) -> Dict[str, Any]:
    """
    Calcula A^{-1} por Gauss–Jordan aplicando operaciones elementales sobre [A | I]
    y registra cada paso. También calcula el determinante |A| a partir de las
    operaciones realizadas:
      - Intercambio de filas: det *= (-1)
      - Escalar fila por α:   det *= α
      - F_i := F_i + c F_j:   det *= 1  (sin cambio)
    Al final, como dejamos a la izquierda la identidad, se cumple:
        1 = det(I) = det(A) * (producto de determinantes de las operaciones)
      => det(A) = 1 / (producto de determinantes de las operaciones).
    Devuelve:
      {
        "estado": "ok" | "no_invertible",
        "Ainv": Matriz | None,
        "pasos": List[str],
        "det": float | None,
        "det_texto": str,  # e.g., "|A| = 5.0000"
        "conclusion": str
      }
    """
    _check_square(A)
    n = len(A)
    # Construir [A | I]
    left = copy_mat(A)
    right = eye(n)

    # Matriz de trabajo n x (2n)
    M = [left[i] + right[i] for i in range(n)]

    pasos: List[str] = []
    def _snap(titulo: str):

        lines = []
        for i in range(n):
            izquierda = " ".join(fmt_number(x, dec) for x in M[i][:n])
            derecha   = " ".join(fmt_number(x, dec) for x in M[i][n:])
            lines.append(f"[ {izquierda} | {derecha} ]")
        pasos.append(titulo + "\n" + "\n".join(lines))

    _snap("Estado inicial: matriz aumentada [A | I]")

    # Para el determinante: acumulamos el determinante de las operaciones aplicadas a A
    det_ops = 1.0
    swaps = 0

    # Procedimiento Gauss–Jordan
    fila = 0
    for col in range(n):
        if fila >= n:
            break

        # 1) Buscar pivote (pivoteo parcial por valor absoluto)
        p = max(range(fila, n), key=lambda r: abs(float(M[r][col])))
        if abs(float(M[p][col])) < tol:
            # Columna sin pivote: A es singular
            _snap(f"No hay pivote en la columna {col+1} (≈0). A no es invertible.")
            conclusion = "La matriz no es invertible porque no tiene pivote en cada fila."
            # determinante es 0 si no hay pivote completo
            return {
                "estado": "no_invertible",
                "Ainv": None,
                "pasos": pasos,
                "det": 0.0,
                "det_texto": f"|A| = {fmt_number(0.0, dec)}",
                "conclusion": conclusion
            }

        if p != fila:
            M[fila], M[p] = M[p], M[fila]
            swaps += 1
            det_ops *= -1.0
            _snap(f"Intercambio de filas F{fila+1} ↔ F{p+1}  (det *= -1)")

        # 2) Normalizar la fila pivote para que el pivote sea 1
        piv = float(M[fila][col])
        if abs(piv) < tol:
            # seguridad adicional (no debería ocurrir tras el máximo)
            _snap(f"Pivote numéricamente nulo en ({fila+1},{col+1}). A no es invertible.")
            return {
                "estado": "no_invertible",
                "Ainv": None,
                "pasos": pasos,
                "det": 0.0,
                "det_texto": f"|A| = {fmt_number(0.0, dec)}",
                "conclusion": "La matriz no es invertible por columna sin pivote."
            }

        inv_piv = 1.0 / piv
        # Escalar fila completa (A y lado derecho)
        for j in range(2*n):
            M[fila][j] *= inv_piv
        det_ops *= inv_piv   # multiplicar fila por inv_piv => det(A) se multiplica por inv_piv
        _snap(f"F{fila+1} := ({fmt_number(inv_piv, dec)}) · F{fila+1}  (det *= {fmt_number(inv_piv, dec)})")

        # 3) Eliminar el resto de entradas en la columna col
        for r in range(n):
            if r == fila:
                continue
            factor = M[r][col]
            if abs(float(factor)) < tol:
                continue
            for j in range(2*n):
                M[r][j] -= factor * M[fila][j]
            _snap(f"F{r+1} := F{r+1} − ({fmt_number(factor, dec)})·F{fila+1}  (det sin cambio)")
        fila += 1

    # Si llegamos aquí, el bloque izquierdo debería ser la identidad
    # Verificación rápida
    ok = True
    for i in range(n):
        for j in range(n):
            ok &= (abs(float(M[i][j]) - (1.0 if i == j else 0.0)) < 10*tol)
    if not ok:
        _snap("La parte izquierda no quedó exactamente como I (inestabilidad numérica).")
        return {
            "estado": "no_invertible",
            "Ainv": None,
            "pasos": pasos,
            "det": 0.0,
            "det_texto": f"|A| = {fmt_number(0.0, dec)}",
            "conclusion": "No se obtuvo [I | A^{-1}] por inestabilidad numérica; A se considera no invertible."
        }

    # Extraer A^{-1} (lado derecho)
    Ainv = [M[i][n:] for i in range(n)]

    # Determinante a partir de las operaciones:
    # det(I) = 1 = det(A) * det_ops  => det(A) = 1 / det_ops
    detA = (1.0 / det_ops) if abs(det_ops) > tol else 0.0
    pasos.append(f"Determinante por operaciones: |A| = 1 / (producto de ops) = {fmt_number(detA, dec)}")

    conclusion = "A es invertible y se obtuvo A^{-1} por Gauss–Jordan."
    pasos.append("Resultado final: [I | A^{-1}]")

    return {
        "estado": "ok",
        "Ainv": Ainv,
        "pasos": pasos,
        "det": detA,
        "det_texto": f"|A| = {fmt_number(detA, dec)}",
        "conclusion": conclusion
    }

# ----------------------------
# Verificación de propiedades (c)(d)(e)
# ----------------------------
def verificar_propiedades_invertibilidad(
    A: Matriz,
    tol: float = DEFAULT_EPS,
    dec: int = 4
) -> Dict[str, Any]:
    """
    Verifica:
      (c) A tiene n pivotes
      (d) Ax = 0 tiene sólo la solución trivial
      (e) Las columnas de A son LI
    Usa los módulos ya existentes para analizar el sistema homogéneo y contar pivotes.
    Devuelve un diccionario con banderas y mensajes interpretativos.
    """
    _check_square(A)
    n = len(A)

    # Analizar Ax = 0
    b0 = [0.0] * n
    info = analizar_sistema(A, b0)   # reutiliza Gauss–Jordan robustecido
    pivotes = info.get("pivotes", [])
    rango = len(pivotes)
    solo_trivial = (info.get("tipo") == "unica")  # en Ax=0, "unica" implica x=0

    # Dependencia/independencia (docente, redundante pero explícito)
    dep = analizar_dependencia([col for col in zip(*A)])  # columnas de A

    prop_c = (rango == n)
    prop_d = bool(solo_trivial)
    prop_e = (dep.get("veredicto") == "independientes")

    mensajes = []
    mensajes.append(("c", prop_c, "Si A tiene n pivotes, entonces A es invertible."))
    mensajes.append(("d", prop_d, "Si Ax = 0 solo tiene la solución trivial, entonces A⁻¹ existe."))
    mensajes.append(("e", prop_e, "Si las columnas son linealmente independientes, entonces A es invertible."))

    # Texto compacto para mostrar en CLI/GUI
    explicacion = ["--- Propiedades teóricas verificadas ---"]
    for clave, ok, interp in mensajes:
        marca = "Se cumple" if ok else "No se cumple"
        explicacion.append(f"({clave}) {marca}. {interp}")

    return {
        "pivotes": pivotes,
        "rango": rango,
        "prop_c": prop_c,
        "prop_d": prop_d,
        "prop_e": prop_e,
        "mensajes": mensajes,
        "explicacion": "\n".join(explicacion),
        "detalle_sistema_homogeneo": info,  # por si quieres mostrar forma paramétrica
        "detalle_dependencia": dep
    }

# ----------------------------
# Atajo integrado: inversa + propiedades
# ----------------------------
def programa_inversa_con_propiedades(
    A: Matriz,
    tol: float = DEFAULT_EPS,
    dec: int = 4
) -> Dict[str, Any]:
    inv = inversa_por_gauss_jordan(A, tol=tol, dec=dec)
    props = verificar_propiedades_invertibilidad(A, tol=tol, dec=dec)
    out = {
        "inversa": inv,
        "propiedades": props
    }
    # Conclusión global
    if inv["estado"] == "ok" and props["prop_c"] and props["prop_d"] and props["prop_e"]:
        out["conclusion_global"] = "A es invertible: se obtuvo A⁻¹ y todas las propiedades (c)(d)(e) se verifican."
    elif inv["estado"] == "ok":
        out["conclusion_global"] = "Se obtuvo A⁻¹, pero alguna propiedad teórica no se verificó (revisar numérico/entrada)."
    else:
        out["conclusion_global"] = "A no es invertible según Gauss–Jordan."
    return out
