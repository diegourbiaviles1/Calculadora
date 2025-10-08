# sistema_lineal.py — Gauss-Jordan con columnas pivote, pasos y salida paramétrica (corregido)
from __future__ import annotations
from typing import List, Dict, Any, Optional
from fractions import Fraction
from utilidad import copy_mat, DEFAULT_EPS, fmt_number, DEFAULT_DEC

class SistemaLineal:
    """
    Resuelve sistemas lineales por Gauss-Jordan (RREF) y muestra las operaciones de filas:
    - Intercambio de filas.
    - Normalización de pivote a 1.
    - Eliminación de coeficientes en la columna pivote.
    """

    def __init__(self, matriz_aumentada: List[List], tol: float = DEFAULT_EPS, decimales: int = 4):
        if not matriz_aumentada or not matriz_aumentada[0]:
            raise ValueError("Matriz aumentada vacía.")
        self.matriz = copy_mat(matriz_aumentada)  # Copia de trabajo
        self.tol = float(tol)
        self.decimales = int(decimales)
        self._k = 0  # Contador de pasos

    # ---------- utilidades internas ----------
    def _is_cero(self, x) -> bool:
        if isinstance(x, Fraction):
            return x == 0
        return abs(float(x)) < self.tol

    def _is_uno(self, x) -> bool:
        if isinstance(x, Fraction):
            return x == 1
        return abs(float(x) - 1.0) < self.tol

    def _fmt_num_local(self, x) -> str:
        if isinstance(x, Fraction):
            return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
        xf = float(x)
        if abs(xf) < 10**(-self.decimales):
            xf = 0.0
        return f"{int(xf)}" if xf.is_integer() else f"{xf:.{self.decimales}f}"

    def _snapshot_matrix(self) -> str:
        filas = len(self.matriz)
        cols  = len(self.matriz[0])
        out = []
        for i in range(filas):
            izq = " ".join(self._fmt_num_local(self.matriz[i][j]) for j in range(cols-1))
            der = self._fmt_num_local(self.matriz[i][cols-1])
            out.append(f"[ {izq} | {der} ]")
        return "\n".join(out)

    def _step(self, descripcion: str) -> str:
        self._k += 1
        header = f"Paso {self._k}: {descripcion}"
        return header + "\n" + self._snapshot_matrix()

    # ---------- algoritmo principal ----------
    def gauss_jordan(self) -> Dict[str, Any]:
        n = len(self.matriz)            # Filas (ecuaciones)
        m = len(self.matriz[0]) - 1     # Columnas de A (variables)
        pasos: List[str] = ["Estado inicial\n" + self._snapshot_matrix()]
        columnas_pivote_1: List[int] = []  # columnas pivote (1-indexed para UI)

        fila = 0
        for col in range(m):
            if fila >= n:
                break

            # 1) Pivoteo parcial
            p = max(range(fila, n), key=lambda r: abs(float(self.matriz[r][col])))
            if self._is_cero(self.matriz[p][col]):
                # Columna sin pivote; variable libre
                continue

            if p != fila:
                self.matriz[fila], self.matriz[p] = self.matriz[p], self.matriz[fila]
                pasos.append(self._step(f"Intercambio F{fila+1} ↔ F{p+1}"))

            # 2) Normalizar la fila pivote
            piv = self.matriz[fila][col]
            if self._is_cero(piv):
                # seguridad ante números muy pequeños
                continue
            inv = (Fraction(1, 1) / piv) if isinstance(piv, Fraction) else 1.0 / float(piv)
            for j in range(m+1):
                self.matriz[fila][j] = self.matriz[fila][j] * inv
            pasos.append(self._step(f"F{fila+1} := ({self._fmt_num_local(inv)})·F{fila+1}"))

            # Registrar pivote (1-indexed)
            columnas_pivote_1.append(col + 1)

            # 3) Eliminar en otras filas
            for r in range(n):
                if r == fila:
                    continue
                factor = self.matriz[r][col]
                if self._is_cero(factor):
                    continue
                for j in range(m+1):
                    self.matriz[r][j] = self.matriz[r][j] - factor * self.matriz[fila][j]
                pasos.append(self._step(f"F{r+1} := F{r+1} − ({self._fmt_num_local(factor)})·F{fila+1}"))

            fila += 1

        # --------- Clasificación del sistema ----------
        # Inconsistente: fila con [0 ... 0 | c] y c != 0
        for i in range(n):
            if all(self._is_cero(self.matriz[i][j]) for j in range(m)) and not self._is_cero(self.matriz[i][m]):
                return {"tipo": "inconsistente", "pasos": pasos, "rref": self.matriz, "pivotes": columnas_pivote_1}

        # Rango (número de columnas con pivote)
        rank = len(set(columnas_pivote_1))

        if rank == m:
            # Única solución
            x = [0.0] * m
            for j in range(m):
                # encontrar fila del 1 en columna j
                i = next(i for i in range(n) if self._is_uno(self.matriz[i][j]))
                x[j] = float(self.matriz[i][m])
            return {"tipo": "unica", "x": x, "pasos": pasos, "rref": self.matriz, "pivotes": columnas_pivote_1}

        # Infinitas soluciones
        pivotes_0 = sorted([c - 1 for c in set(columnas_pivote_1)])  # 0-index para cálculo
        libres_0 = [j for j in range(m) if j not in pivotes_0]
        libres_1 = [j + 1 for j in libres_0]  # devolver 1-indexed como en UI

        # Solución particular (libres = 0)
        x_part = [0.0] * m
        for j in pivotes_0:
            i = next(i for i in range(n) if self._is_uno(self.matriz[i][j]))
            x_part[j] = float(self.matriz[i][m])

        # Base del espacio nulo
        base_nulo: List[List[float]] = []
        for ell in libres_0:
            vec = [0.0] * m
            vec[ell] = 1.0
            for j in pivotes_0:
                i = next(i for i in range(n) if self._is_uno(self.matriz[i][j]))
                coef = self.matriz[i][ell]           # <-- corregido (antes había 'matrizd')
                if not self._is_cero(coef):
                    vec[j] = -float(coef)
            base_nulo.append(vec)

        return {
            "tipo": "infinitas",
            "libres": libres_1,            # columnas libres (1-indexed)
            "x_part": x_part,              # solución particular (libres = 0)
            "base_nulo": base_nulo,        # base del espacio nulo (lista de vectores)
            "pasos": pasos,
            "rref": self.matriz,
            "pivotes": columnas_pivote_1   # 1-indexed
        }

# -------- Formato de solución paramétrica --------
def formatear_solucion_parametrica(
    out: Dict[str, Any],
    nombres_vars: Optional[list[str]] = None,
    dec: int = DEFAULT_DEC,
    fracciones: bool = True
) -> str:
    """
    Genera texto en forma paramétrica a partir del resultado de gauss_jordan():
    - Variables libres y asignaciones explícitas.
    - Forma vectorial: x = x_part + Σ s_k·w_k.
    - Indicador de soluciones triviales o no triviales.
    """
    if out.get("tipo") == "inconsistente":
        return "Sistema inconsistente: no existe forma paramétrica."
    if "rref" not in out or "pivotes" not in out:
        return "No hay datos suficientes para formar la solución paramétrica."

    R = out["rref"]
    n = len(R)
    m = len(R[0]) - 1
    piv1 = out.get("pivotes", [])              # 1-indexed
    piv0 = sorted([p - 1 for p in piv1])       # 0-indexed
    libres0 = sorted([j for j in range(m) if j not in piv0])

    # nombres de variables
    if not nombres_vars or len(nombres_vars) != m:
        nombres_vars = [f"x{j+1}" for j in range(m)]

    # mapeo libre -> símbolo s_k
    s_names = [f"s{idx+1}" for idx in range(len(libres0))]
    libre_to_s = {ell: s_names[k] for k, ell in enumerate(libres0)}

    lineas: List[str] = []
    lineas.append("Solución general (forma paramétrica):")

    # 1) Variables libres
    if libres0:
        for idx, ell in enumerate(libres0):
            lineas.append(f"  {nombres_vars[ell]} = {s_names[idx]}")
    else:
        lineas.append("  (no hay variables libres)")

    # 2) Asignación de variables pivote
    for j in piv0:
        i = next(i for i in range(n) if (R[i][j] == 1) or (abs(float(R[i][j]) - 1.0) < DEFAULT_EPS))
        b_val = float(R[i][m])
        rhs_terms: List[str] = []
        if abs(b_val) > DEFAULT_EPS:  # evita agregar 0
            rhs_terms.append(fmt_number(b_val, dec, fracciones))
        for ell in libres0:
            coef = float(R[i][ell])
            if abs(coef) > DEFAULT_EPS:
                s = libre_to_s[ell]
                sgn = " - " if coef >= 0 else " + "
                rhs_terms.append(f"{sgn}{fmt_number(abs(coef), dec, fracciones)}{s}")
        rhs = " ".join(rhs_terms) if rhs_terms else "0"
        lineas.append(f"  {nombres_vars[j]} = {rhs}")

    # 3) Forma vectorial (si aplica)
    x_part = out.get("x_part")
    base_nulo = out.get("base_nulo")
    if x_part is not None and base_nulo is not None:
        lineas.append("\n• Forma vectorial:")
        xp = "[" + "  ".join(fmt_number(float(x), dec, fracciones) for x in x_part) + "]"
        if base_nulo:
            partes = []
            for k, vec in enumerate(base_nulo):
                vtxt = "[" + "  ".join(fmt_number(float(x), dec, fracciones) for x in vec) + "]"
                partes.append(f"{s_names[k]}·{vtxt}")
            lineas.append(f"  x = {xp}  +  " + "  +  ".join(partes))
        else:
            lineas.append(f"  x = {xp}")

        # Indicador trivial / no trivial
        if all(abs(float(x)) < 1e-12 for x in x_part) and not base_nulo:
            lineas.append("\nSolución trivial: solo x = 0.")
        elif all(abs(float(x)) < 1e-12 for x in x_part) and base_nulo:
            lineas.append("\nSoluciones no triviales: existen infinitas soluciones distintas de x = 0.")

    return "\n".join(lineas)
