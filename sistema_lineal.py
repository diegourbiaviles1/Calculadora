# sistema_lineal.py
from typing import List, Dict, Any
from utilidad import copy_mat

class SistemaLineal:
    """
    Resuelve sistemas lineales por Gauss-Jordan (RREF).
    Mantiene un registro de 'pasos' legibles para explicar el procedimiento.
    """
    def __init__(self, matriz_aumentada: List[List[float]], tol: float = 1e-10, decimales: int = 4):
        if not matriz_aumentada or not matriz_aumentada[0]:
            raise ValueError("Matriz aumentada vacía.")
        self.matriz = copy_mat(matriz_aumentada)  # se modifica in-place durante el proceso
        self.tol = float(tol)
        self.decimales = int(decimales)

    # ---------- utilidades internas ----------
    def _is_cero(self, x: float) -> bool:
        return abs(x) < self.tol

    def _fmt(self, x: float) -> str:
        if self._is_cero(x):
            x = 0.0
        return f"{int(x)}" if float(x).is_integer() else f"{x:.{self.decimales}f}"

    def _snapshot(self, titulo: str) -> str:
        filas = len(self.matriz)
        cols = len(self.matriz[0])
        lines = [titulo, "-" * max(12, len(titulo))]
        for i in range(filas):
            izq = " ".join(self._fmt(self.matriz[i][j]) for j in range(cols - 1))
            der = self._fmt(self.matriz[i][cols - 1])
            lines.append(f"[ {izq} | {der} ]")
        return "\n".join(lines)

    # ---------- algoritmo principal ----------
    def gauss_jordan(self) -> Dict[str, Any]:
        n = len(self.matriz)            # filas (ecuaciones)
        m = len(self.matriz[0]) - 1     # variables (columnas de A)
        pasos: List[str] = [self._snapshot("Inicial")]

        fila = 0
        for col in range(m):
            if fila >= n:
                break

            # pivoteo parcial: escoger fila con mayor |coef|
            p = max(range(fila, n), key=lambda r: abs(self.matriz[r][col]))
            if self._is_cero(self.matriz[p][col]):
                # columna sin pivote útil → se queda como libre
                continue

            if p != fila:
                self.matriz[fila], self.matriz[p] = self.matriz[p], self.matriz[fila]
                pasos.append(self._snapshot(f"Permutar filas {fila+1} ↔ {p+1}"))

            piv = self.matriz[fila][col]

            # normalizar fila del pivote
            for j in range(m + 1):
                self.matriz[fila][j] /= piv
            pasos.append(self._snapshot(f"Normalizar fila {fila+1} (÷ {self._fmt(piv)})"))

            # anular el resto en la columna del pivote
            for r in range(n):
                if r == fila:
                    continue
                factor = self.matriz[r][col]
                if self._is_cero(factor):
                    continue
                for j in range(m + 1):
                    self.matriz[r][j] -= factor * self.matriz[fila][j]
                pasos.append(self._snapshot(f"F{r+1} = F{r+1} - ({self._fmt(factor)})·F{fila+1}"))

            fila += 1

        # --------- Clasificación del sistema ----------
        # Inconsistente: fila con [0 ... 0 | c] y c != 0
        for i in range(n):
            if all(self._is_cero(self.matriz[i][j]) for j in range(m)) and not self._is_cero(self.matriz[i][m]):
                return {"tipo": "inconsistente", "pasos": pasos, "rref": self.matriz}

        # Rango y columnas pivote
        rank = 0
        columnas_pivote = []
        for j in range(m):
            fila_con_uno = None
            ok_col = True
            for i in range(n):
                x = self.matriz[i][j]
                if abs(x - 1.0) < self.tol:
                    if fila_con_uno is None:
                        fila_con_uno = i
                    else:
                        ok_col = False
                elif not self._is_cero(x):
                    ok_col = False
            if ok_col and fila_con_uno is not None:
                rank += 1
                columnas_pivote.append(j)

        # Única solución: rank = m
        if rank == m:
            x = [0.0] * m
            for j in range(m):
                i = next(i for i in range(n) if abs(self.matriz[i][j] - 1.0) < self.tol)
                x[j] = self.matriz[i][m]
            return {"tipo": "unica", "x": x, "pasos": pasos, "rref": self.matriz}

        # Infinitas soluciones: variables libres = columnas no pivote
        libres = [j for j in range(m) if j not in columnas_pivote]
        return {"tipo": "infinitas", "libres": libres, "pasos": pasos, "rref": self.matriz}
