
class SistemaLineal:
    def __init__(self, matriz_aumentada, tol=1e-10, decimales=2):
        # Copia por valor (lista de listas)
        self.matriz = [fila[:] for fila in matriz_aumentada]
        self.tol = float(tol)
        self.decimales = int(decimales)

    def _is_cero(self, x):
        return abs(x) < self.tol

    def _fmt(self, x):
        # Evita -0.00 y muestra enteros sin .00
        if self._is_cero(x):
            x = 0.0
        return f"{int(x)}" if float(x).is_integer() else f"{x:.{self.decimales}f}"

    def _snapshot(self, paso, operacion):
        # String del estado actual de la matriz
        filas_txt = [ "  ".join(self._fmt(v) for v in fila) for fila in self.matriz ]
        return f"Paso {paso} ({operacion}):\n" + "\n".join(filas_txt) + "\n"

    def eliminacion_gaussiana(self):
        """
        Aplica Gauss-Jordan (reduce completamente) y devuelve:
        - Traza de pasos
        - Interpretación de soluciones
        """
        if not self.matriz or not self.matriz[0]:
            return "Matriz no válida."

        filas, columnas = len(self.matriz), len(self.matriz[0])
        ult_col = columnas - 1

        paso = 1
        pasos = []

        fila_actual = 0
        for col in range(ult_col):
            if fila_actual >= filas:
                break

            # Pivoteo parcial: fila con mayor |coef|
            max_row = max(range(fila_actual, filas), key=lambda i: abs(self.matriz[i][col]))
            if self._is_cero(self.matriz[max_row][col]):
                # Columna sin pivote útil
                continue

            # Intercambio si el pivote no está ya en fila_actual
            if max_row != fila_actual:
                self.matriz[fila_actual], self.matriz[max_row] = self.matriz[max_row], self.matriz[fila_actual]
                pasos.append(self._snapshot(paso, f"Intercambio f{fila_actual+1} <-> f{max_row+1}"))
                paso += 1

            # Normalizar fila pivote
            pivote = self.matriz[fila_actual][col]
            if not self._is_cero(pivote) and pivote != 1.0:
                inv = 1.0 / pivote
                fila_piv = self.matriz[fila_actual]
                for k in range(col, columnas):
                    fila_piv[k] *= inv
                pasos.append(self._snapshot(paso, f"f{fila_actual+1} -> (1/{self._fmt(pivote)}) * f{fila_actual+1}"))
                paso += 1

            # Eliminar por encima y por debajo (Gauss-Jordan)
            for i in range(filas):
                if i == fila_actual:
                    continue
                factor = self.matriz[i][col]
                if self._is_cero(factor):
                    continue
                fila_i = self.matriz[i]
                fila_piv = self.matriz[fila_actual]
                for k in range(col, columnas):
                    fila_i[k] -= factor * fila_piv[k]
                pasos.append(self._snapshot(paso, f"f{i+1} -> f{i+1} - ({self._fmt(factor)}) * f{fila_actual+1}"))
                paso += 1

            fila_actual += 1

        # Interpretación
        interpretacion = self.interpretar_resultado()
        pasos.append(interpretacion)
        return "\n".join(pasos)

    def interpretar_resultado(self):
        """
        Informa: inconsistencia / únicas / infinitas soluciones.
        Da ecuaciones xj = ...
        """
        n = len(self.matriz)
        m = len(self.matriz[0]) - 1
        A = self.matriz

        # Detectar inconsistencia: [0 ... 0 | c] con c != 0
        for i in range(n):
            if all(self._is_cero(A[i][j]) for j in range(m)) and not self._is_cero(A[i][m]):
                return "Solución del sistema:\n\nEl sistema es inconsistente y no tiene soluciones.\n"

        # Encontrar columnas pivote (buscando 1 "limpio" y ceros en la columna)
        pivote_en_col = [-1] * m   # fila del pivote por columna o -1
        columnas_pivote = []
        for j in range(m):
            fila_1 = -1
            ok = True
            for i in range(n):
                if abs(A[i][j] - 1.0) < self.tol:
                    if fila_1 == -1:
                        fila_1 = i
                    else:
                        ok = False  # dos unos en misma columna
                        break
                elif not self._is_cero(A[i][j]):
                    ok = False
                    break
            if ok and fila_1 != -1:
                pivote_en_col[j] = fila_1
                columnas_pivote.append(j+1)

        # Variables libres: columnas sin pivote
        libres = [j for j in range(m) if pivote_en_col[j] == -1]

        # Construir ecuaciones x_j = c + sum(coef * x_libre)
        lineas = ["Solución del sistema:"]
        soluciones_numericas = {}
        for j in range(m):
            var = f"x{j+1}"
            if pivote_en_col[j] == -1:
                lineas.append(f"{var} es libre")
                continue

            fila = pivote_en_col[j]
            c = A[fila][m]
            partes = []
            # constante
            const_str = self._fmt(c)
            if not self._is_cero(c):
                partes.append(const_str)

            # términos por variables libres (signo cambia al pasar a la derecha)
            for k in libres:
                coef = -A[fila][k]
                if self._is_cero(coef):
                    continue
                s = self._fmt(coef) + f"x{k+1}"
                # Coloca signo explícito para concatenar
                if coef > 0 and partes:
                    s = "+ " + s
                partes.append(s)

            if not partes:
                expr = "0"
                soluciones_numericas[var] = 0.0
            else:
                expr = " ".join(partes).lstrip("+ ").strip()
                # si no depende de libres, guarda valor numérico
                if not libres:
                    soluciones_numericas[var] = c

            lineas.append(f"{var} = {expr}")

        # Diagnóstico final
        if libres:
            lineas.append("\nHay infinitas soluciones debido a variables libres.")
        else:
            # Única solución (todas con pivote)
            if len(soluciones_numericas) == m and all(self._is_cero(v) for v in soluciones_numericas.values()):
                lineas.append("\nSolución trivial.")
            else:
                lineas.append("\nLa solución es única.")

        # Info extra: columnas pivote
        if columnas_pivote:
            lineas.append(f"\nLas columnas pivote son: {', '.join(map(str, columnas_pivote))}.")
        else:
            lineas.append("\nNo hay columnas pivote.")

        return "\n".join(lineas)

    # ---------- IO de consola (sin append, robusto) ----------
    @staticmethod
    def leer_matriz_desde_teclado():
        while True:
            try:
                m = int(input("Número de filas: ").strip())
                n = int(input("Número de columnas: ").strip())
                if m <= 0 or n <= 0:
                    print("Por favor, ingresa valores positivos.")
                    continue
                break
            except ValueError:
                print("Entrada no válida. Intenta de nuevo.")

        print("\nIntroduce cada fila de la MATRIZ AUMENTADA con", n + 1, "valores separados por espacio.")
        print("Ejemplo (n=3):  2 -1 3 5   ← equivale a [2, -1, 3 | 5]\n")

        matriz = []
        for i in range(m):
            partes = input(f"Fila {i+1}: ").replace("|", " ").split()
            if len(partes) != n + 1:
                raise ValueError(f"La fila {i+1} no tiene {n+1} números.")
            try:
                fila = [float(x) for x in partes]
            except ValueError:
                raise ValueError(f"Entrada inválida en fila {i+1}.")
            matriz.append(fila)

        return SistemaLineal(matriz)

    @staticmethod
    def resolver_desde_teclado():
        sistema = SistemaLineal.leer_matriz_desde_teclado()
        print("\n===Resolviendo por Eliminación Gaussiana (con pasos)===")
        print(sistema.eliminacion_gaussiana())


if __name__ == "__main__":
    SistemaLineal.resolver_desde_teclado()