# sistema_lineal.py — Gauss-Jordan con columnas pivote y operaciones de fila detalladas
from typing import List, Dict, Any
from fractions import Fraction
from utilidad import copy_mat

class SistemaLineal:
    """
    Resuelve sistemas lineales por Gauss-Jordan (RREF) y muestra todas las operaciones de filas:
    - Intercambio de filas.
    - Normalización de fila (hacer pivote igual a 1).
    - Eliminación de los coeficientes debajo y encima del pivote.
    """
    def __init__(self, matriz_aumentada: List[List], tol: float = 1e-10, decimales: int = 4):
        if not matriz_aumentada or not matriz_aumentada[0]:
            raise ValueError("Matriz aumentada vacía.")
        self.matriz = copy_mat(matriz_aumentada)  # Copia de la matriz original
        self.tol = float(tol)
        self.decimales = int(decimales)
        self._k = 0  # Contador de pasos

    # ---------- utilidades internas ----------
    def _is_cero(self, x) -> bool:
        """Verifica si un valor es esencialmente cero considerando la tolerancia."""
        if isinstance(x, Fraction):
            return x == 0
        return abs(float(x)) < self.tol

    def _fmt_num(self, x) -> str:
        """Formatea el número como fracción si es necesario o en decimal con 4 decimales."""
        if isinstance(x, Fraction):
            return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
        xf = float(x)
        if abs(xf) < 10**(-self.decimales):
            xf = 0.0
        return f"{int(xf)}" if xf.is_integer() else f"{xf:.{self.decimales}f}"

    def _snapshot_matrix(self) -> str:
        """Genera un snapshot de la matriz actual en formato amigable para imprimir."""
        filas = len(self.matriz)
        cols  = len(self.matriz[0])
        out = []
        for i in range(filas):
            izq = " ".join(self._fmt_num(self.matriz[i][j]) for j in range(cols-1))
            der = self._fmt_num(self.matriz[i][cols-1])
            out.append(f"[ {izq} | {der} ]")
        return "\n".join(out)

    def _step(self, descripcion: str) -> str:
        """Registra un paso con su descripción y la matriz actualizada."""
        self._k += 1
        header = f"Paso {self._k}: {descripcion}"
        return header + "\n" + self._snapshot_matrix()

    # ---------- algoritmo principal ----------
    def gauss_jordan(self) -> Dict[str, Any]:
        """Resuelve el sistema usando el método de Gauss-Jordan y muestra todas las operaciones de filas."""
        n = len(self.matriz)            # Filas (ecuaciones)
        m = len(self.matriz[0]) - 1     # Columnas de A (variables)
        pasos: List[str] = ["Estado inicial\n" + self._snapshot_matrix()]
        columnas_pivote = []            # Lista para las columnas pivote

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

            # Registrar la columna pivote
            columnas_pivote.append(col + 1)  # Guardamos la columna pivote (1-indexed)

            # 3) Eliminar en las filas restantes (encima y debajo del pivote)
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
                return {"tipo": "inconsistente", "pasos": pasos, "rref": self.matriz, "pivotes": columnas_pivote}

        # Rango y columnas pivote
        rank = 0
        columnas_pivote = sorted(set(columnas_pivote))  # Eliminar duplicados y ordenar
        for j in range(m):
            fila_con_uno = None
            ok_col = True
            for i in range(n):
                x = self.matriz[i][j]
                if abs(float(x) - 1.0) < self.tol if not isinstance(x, Fraction) else (x == 1):
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
            return {"tipo": "unica", "x": x, "pasos": pasos, "rref": self.matriz, "pivotes": columnas_pivote}

        # Infinitas soluciones
        libres = [j for j in range(m) if j not in columnas_pivote]
        return {"tipo": "infinitas", "libres": libres, "pasos": pasos, "rref": self.matriz, "pivotes": columnas_pivote}
