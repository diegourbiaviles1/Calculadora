# sistema_lineal.py — Gauss-Jordan con columnas pivote, pasos y salida paramétrica
from __future__ import annotations
from typing import List, Dict, Any, Optional
from fractions import Fraction
from utilidad import copy_mat, DEFAULT_EPS, fmt_number, DEFAULT_DEC

class SistemaLineal:
    """
    Resuelve sistemas lineales por Gauss-Jordan (RREF) y muestra todas las operaciones de filas:
    - Intercambio de filas.
    - Normalización de fila (hacer pivote igual a 1).
    - Eliminación de coeficientes encima y debajo del pivote.
    """
    def __init__(self, matriz_aumentada: List[List], tol: float = DEFAULT_EPS, decimales: int = 4):
        if not matriz_aumentada or not matriz_aumentada[0]:
            raise ValueError("Matriz aumentada vacía.")
        self.matriz = copy_mat(matriz_aumentada)  # Copia de la matriz original
        self.tol = float(tol)
        self.decimales = int(decimales)
        self._k = 0  # Contador de pasos

    # ---------- utilidades internas ----------
    def _is_cero(self, x) -> bool:
        if isinstance(x, Fraction):
            return x == 0
        return abs(float(x)) < self.tol

    def _fmt_num(self, x) -> str:
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
            izq = " ".join(self._fmt_num(self.matriz[i][j]) for j in range(cols-1))
            der = self._fmt_num(self.matriz[i][cols-1])
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
        columnas_pivote_1 = []          # columnas pivote (1-indexed para UI)

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
            inv = (Fraction(1, 1) / piv) if isinstance(piv, Fraction) else 1.0 / float(piv)
            for j in range(m+1):
                self.matriz[fila][j] *= inv
            pasos.append(self._step(f"F{fila+1} := ({self._fmt_num(inv)})·F{fila+1}"))

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
                    self.matriz[r][j] -= factor * self.matriz[fila][j]
                pasos.append(self._step(f"F{r+1} := F{r+1} − ({self._fmt_num(factor)})·F{fila+1}"))

            fila += 1

        # --------- Clasificación del sistema ----------
        # Inconsistente: fila con [0 ... 0 | c] y c != 0
        for i in range(n):
            if all(self._is_cero(self.matriz[i][j]) for j in range(m)) and not self._is_cero(self.matriz[i][m]):
                return {"tipo": "inconsistente", "pasos": pasos, "rref": self.matriz, "pivotes": columnas_pivote_1}

        # Rango (contar columnas que tienen un único 1 y resto 0)
        rank = 0
        for j in range(m):
            fila_con_uno = None
            ok_col = True
            for i in range(n):
                x = self.matriz[i][j]
                if (x == 1) if isinstance(x, Fraction) else (abs(float(x) - 1.0) < self.tol):
                    if fila_con_uno is None:
                        fila_con_uno = i
                    else:
                        ok_col = False
                elif not self._is_cero(x):
                    ok_col = False
            if ok_col and fila_con_uno is not None:
                rank += 1

        if rank == m:
            # Única solución
            x = [0] * m
            for j in range(m):
                i = next(i for i in range(n)
                         if (self.matriz[i][j] == 1 if isinstance(self.matriz[i][j], Fraction)
                             else abs(float(self.matriz[i][j]) - 1.0) < self.tol))
                x[j] = self.matriz[i][m]
            return {"tipo": "unica", "x": x, "pasos": pasos, "rref": self.matriz, "pivotes": columnas_pivote_1}

        # Infinitas soluciones
        pivotes_0 = sorted([c - 1 for c in set(columnas_pivote_1)])  # 0-index para cálculo
        libres_0 = [j for j in range(m) if j not in pivotes_0]
        libres_1 = [j + 1 for j in libres_0]  # devolver 1-indexed como en UI

        # Solución particular (libres = 0)
        x_part = [0.0] * m
        for j in pivotes_0:
            i = next(i for i in range(n)
                     if (self.matriz[i][j] == 1 if isinstance(self.matriz[i][j], Fraction)
                         else abs(float(self.matriz[i][j]) - 1.0) < self.tol))
            x_part[j] = float(self.matriz[i][m])

        # Base del espacio nulo
        base_nulo = []
        for ell in libres_0:
            vec = [0.0] * m
            vec[ell] = 1.0
            for j in pivotes_0:
                i = next(i for i in range(n)
                         if (self.matriz[i][j] == 1 if isinstance(self.matriz[i][j], Fraction)
                             else abs(float(self.matriz[i][j]) - 1.0) < self.tol))
                coef = self.matriz[i][ell]
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

# -------- Formato de solución paramétrica (para imprimir como en la foto) --------
def formatear_solucion_parametrica(
    out: Dict[str, Any],
    nombres_vars: Optional[list[str]] = None,
    dec: int = DEFAULT_DEC,
    fracciones: bool = True
) -> str:
    """
    Genera texto en forma paramétrica a partir del resultado de gauss_jordan():
    - Ecuaciones implícitas tipo: x_j + sum(c_ell * x_ell) = b
    - Variables libres: x_ell = s_k
    - Asignación explícita: x_j = b - sum(c_ell * s_k)
    - Forma vectorial: x = x_part + Σ s_k·w_k (si x_part y base_nulo existen)

    out debe contener: 'rref', 'pivotes' (1-indexed) y opcionalmente 'libres','x_part','base_nulo'.
    """
    if "rref" not in out or "pivotes" not in out:
        return "No hay datos suficientes para formar la solución paramétrica."

    R = out["rref"]
    n = len(R)            # filas
    m = len(R[0]) - 1     # variables
    piv1 = out.get("pivotes", [])              # 1-indexed
    piv0 = sorted([p - 1 for p in piv1])       # 0-indexed
    libres0 = sorted([j for j in range(m) if j not in piv0])
    libres1 = [j + 1 for j in libres0]

    # nombres de variables
    if not nombres_vars or len(nombres_vars) != m:
        nombres_vars = [f"x{j+1}" for j in range(m)]

    # mapeo libre -> símbolo s_k
    s_names = [f"s{idx+1}" for idx in range(len(libres0))]
    libre_to_s = {ell: s_names[k] for k, ell in enumerate(libres0)}

    lineas = []
    lineas.append("Solución general (forma paramétrica):\n")

    # 1) Ecuaciones implícitas tipo: xj + Σ c_ell x_ell = b
    lineas.append("• Ecuaciones del RREF:")
    for j in piv0:
        # encontrar la fila con el 1 en la columna j
        i = next(i for i in range(n) if (R[i][j] == 1) or (abs(float(R[i][j]) - 1.0) < 1e-12))
        terms = [nombres_vars[j]]
        for ell in libres0:
            coef = float(R[i][ell])
            if abs(coef) > 1e-12:
                sgn = " + " if coef >= 0 else " - "
                terms.append(f"{sgn}{fmt_number(abs(coef), dec, fracciones)}{nombres_vars[ell]}")
        b = float(R[i][m])
        eq = " ".join(terms) + f" = {fmt_number(b, dec, fracciones)}"
        lineas.append("  " + eq)

    # 2) Declaración de libres
    lineas.append("\n• Variables libres:")
    if libres0:
        for idx, ell in enumerate(libres0):
            lineas.append(f"  {nombres_vars[ell]} = {s_names[idx]}")
    else:
        lineas.append("  (no hay)")

    # 3) Asignación explícita por variables (xj en función de s_k)
    lineas.append("\n• Asignación paramétrica:")
    for j in piv0:
        i = next(i for i in range(n) if (R[i][j] == 1) or (abs(float(R[i][j]) - 1.0) < 1e-12))
        rhs_terms = [fmt_number(float(R[i][m]), dec, fracciones)]
        for ell in libres0:
            coef = float(R[i][ell])
            if abs(coef) > 1e-12:
                s = libre_to_s[ell]
                sgn = " - " if coef >= 0 else " + "
                rhs_terms.append(f"{sgn}{fmt_number(abs(coef), dec, fracciones)}{s}")
        rhs = " ".join(rhs_terms)
        lineas.append(f"  {nombres_vars[j]} = {rhs}")

    for ell in libres0:
        lineas.append(f"  {nombres_vars[ell]} = {libre_to_s[ell]}")

    # 4) Forma vectorial (si hay datos)
    x_part = out.get("x_part")
    base_nulo = out.get("base_nulo")
    if x_part is not None and base_nulo is not None:
        lineas.append("\n• Forma vectorial:")
        xp = "[" + "  ".join(fmt_number(float(x), dec, fracciones) for x in x_part) + "]^T"
        if base_nulo:
            partes = []
            for k, vec in enumerate(base_nulo):
                vtxt = "[" + "  ".join(fmt_number(float(x), dec, fracciones) for x in vec) + "]^T"
                partes.append(f"{s_names[k]}·{vtxt}")
            lineas.append(f"  x = {xp}  +  " + "  +  ".join(partes))
        else:
            lineas.append(f"  x = {xp}")

    return "\n".join(lineas)
