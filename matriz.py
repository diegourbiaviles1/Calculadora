class SistemaLineal:
    def __init__(self, matriz_aumentada):
        # Guardamos la matriz aumentada tal cual la recibe el programa
        # (última columna = término independiente)
        self.matriz = [fila[:] for fila in matriz_aumentada]

    # -------------------------------
    # TUS FUNCIONES (con comentarios)
    # -------------------------------
    def imprimir_matriz(self, paso, operacion):
        """
        Crea una representación string de la matriz en un formato legible.
        :param paso: número del paso actual en el proceso de eliminación
        :param operacion: descripción de la operación realizada
        :return: string que representa el estado actual de la matriz
        """
        # Encabezado: muestra el número de paso y la operación elemental aplicada
        texto = f"Paso {paso} ({operacion}):\n"
        # Recorremos cada fila y la mostramos con 2 decimales para seguir la evolución de la matriz
        for fila in self.matriz:
            texto += "  ".join(f"{valor:.2f}" for valor in fila) + "\n"
        # Línea en blanco para separar visualmente un paso del siguiente
        texto += "\n"
        return texto

    def eliminacion_gaussiana(self):
        """
        Aplica el método de eliminación Gaussiana para transformar la matriz a su forma escalonada.
        :return: string con todos los pasos de la eliminación y la solución interpretada
        """
        # Comprobación básica: si no hay datos, no podemos proceder
        if not self.matriz:
            return "Matriz no válida."

        paso = 1
        resultado = ""
        filas, columnas = len(self.matriz), len(self.matriz[0])  # Determina dimensiones (m x (n+1))
        fila_actual = 0  # En qué fila intentaremos colocar el siguiente pivote

        # Recorremos columnas de variables (dejamos fuera la columna del término independiente)
        for col in range(columnas - 1):
            if fila_actual >= filas:
                break  # Si ya usamos todas las filas posibles para pivotes, terminamos

            # 1) Pivoteo parcial: elegimos como pivote el valor con mayor |·| en esta columna
            max_row = max(range(fila_actual, filas), key=lambda i: abs(self.matriz[i][col]))
            if abs(self.matriz[max_row][col]) < 1e-10:
                # Si toda la columna (desde fila_actual hacia abajo) es ~0, no sirve para pivotear
                continue

            # 2) Si el mejor pivote no está en la fila_actual, intercambiamos filas
            if fila_actual != max_row:
                self.matriz[fila_actual], self.matriz[max_row] = self.matriz[max_row], self.matriz[fila_actual]
                resultado += self.imprimir_matriz(paso, f"Intercambio f{fila_actual + 1} <-> f{max_row + 1}")
                paso += 1

            pivote = self.matriz[fila_actual][col]  # Valor del pivote

            # 3) Normalizamos la fila del pivote para que el pivote sea exactamente 1
            if abs(pivote) > 1e-10:
                self.matriz[fila_actual] = [elemento / pivote for elemento in self.matriz[fila_actual]]
                resultado += self.imprimir_matriz(paso, f"f{fila_actual + 1} -> (1/{pivote:.2f}) * f{fila_actual + 1}")
                paso += 1

            # 4) Eliminamos el resto de la columna (hacemos 0 en todas las otras filas)
            for i in range(filas):
                if i != fila_actual:
                    factor = self.matriz[i][col]
                    if abs(factor) > 1e-10:
                        # f_i := f_i - factor * f_pivote
                        self.matriz[i] = [
                            self.matriz[i][k] - factor * self.matriz[fila_actual][k] for k in range(columnas)
                        ]
                        resultado += self.imprimir_matriz(paso, f"f{i + 1} -> f{i + 1} - ({factor:.2f}) * f{fila_actual + 1}")
                        paso += 1

            # Avanzamos de fila para buscar el siguiente pivote en la siguiente columna
            fila_actual += 1

        # 5) Con la matriz ya reducida por filas, interpretamos la (las) solución(es)
        resultado += self.interpretar_resultado()
        return resultado

    def interpretar_resultado(self):
        """
        Interpreta la matriz escalonada reducida, indicando si el sistema es consistente, y hallando las soluciones.
        :return: string que describe la solución del sistema y las columnas pivote
        """
        # n = filas (ecuaciones), m = columnas de variables (sin contar término independiente)
        n, m = len(self.matriz), len(self.matriz[0]) - 1
        pivotes = [-1] * m  # pivotes[j] = fila donde x_{j+1} tiene 1 "limpio", -1 si la variable es libre
        resultado = "Solución del sistema:\n"
        soluciones = {}            # Texto por variable: valor, libre o inconsistente
        soluciones_numericas = {}  # Valores numéricos cuando no dependen de libres
        columnas_pivote = []       # Para mostrar qué columnas tuvieron pivote (1-based)

        # (1) Detectar columnas pivote buscando un 1 aislado en cada columna
        for j in range(m):
            for i in range(n):
                if abs(self.matriz[i][j] - 1) < 1e-10 and all(
                        abs(self.matriz[k][j]) < 1e-10 for k in range(n) if k != i):
                    pivotes[j] = i
                    columnas_pivote.append(j + 1)
                    break

        # (2) Buscar filas del tipo [0 0 ... 0 | c] con c != 0 → inconsistente
        fila_inconsistente = [
            i for i, fila in enumerate(self.matriz)
            if all(abs(val) < 1e-10 for val in fila[:-1]) and abs(fila[-1]) > 1e-10
        ]
        # Marcador de inconsistencia (si existe, el sistema no tiene solución)
        inconsistente_var = set(f"x{i + 1}" for i in fila_inconsistente)

        # (3) Construir expresiones de cada variable x_j
        for j in range(m):
            var_name = f"x{j + 1}"
            if var_name in inconsistente_var:
                # Señalamos explícitamente la inconsistencia
                soluciones[var_name] = f"{var_name} es inconsistente"
            elif pivotes[j] == -1:
                # Sin pivote → variable libre (parámetro)
                soluciones[var_name] = f"{var_name} es libre"
            else:
                # Con pivote → x_j = constante ± (coef)*variables_libres
                fila = pivotes[j]
                constante = self.matriz[fila][-1]
                # Si la constante es entera, sin decimales; de lo contrario, dos decimales
                constante_str = (f"{int(constante)}" if float(constante).is_integer() else f"{constante:.2f}")
                terminos = []
                # Recorrer otras columnas buscando coeficientes de variables libres en la misma fila
                for k in range(m):
                    if k != j and pivotes[k] == -1 and abs(self.matriz[fila][k]) > 1e-10:
                        # Al pasar al otro lado cambia el signo
                        coef = -self.matriz[fila][k]
                        coef_str = (f"{int(coef)}" if float(coef).is_integer() else f"{coef:.2f}")
                        # Construimos “+/- coef * x_k”
                        if coef < 0:
                            terminos.append(f"{coef_str}x{k + 1}")
                        else:
                            terminos.append(f"+ {coef_str}x{k + 1}")

                # Armamos “x_j = <constante> ± ...” con cuidado de los signos y el caso 0
                if constante_str == "0" and not terminos:
                    ecuacion = "0"
                else:
                    ecuacion = ""
                    if constante_str != "0":
                        ecuacion += constante_str
                    if terminos:
                        if ecuacion and ecuacion != "0":
                            ecuacion += " " + " ".join(terminos)
                        else:
                            ecuacion = " ".join(terminos).lstrip("+ ").strip()

                soluciones[var_name] = f"{var_name} = {ecuacion}".strip()

                # Si no depende de libres, guardamos valor numérico
                if not terminos:
                    soluciones_numericas[var_name] = constante

        # (4) Volcado ordenado de x1, x2, ..., xm
        for i in range(m):
            var_name = f"x{i + 1}"
            if var_name in soluciones:
                resultado += f"{soluciones[var_name]}\n"

        # (5) Diagnóstico final del sistema
        if inconsistente_var:
            resultado += "\nEl sistema es inconsistente y no tiene soluciones.\n"
        elif any(pivote == -1 for pivote in pivotes):
            resultado += "\nHay infinitas soluciones debido a variables libres.\n"
        else:
            # Todas las variables fijadas: si todas son 0, lo indicamos como “trivial”
            if len(soluciones_numericas) == m and all(abs(val) < 1e-10 for val in soluciones_numericas.values()):
                resultado += "\nSolución trivial.\n"
            else:
                resultado += "\nLa solución es única.\n"

        # (6) Información complementaria: cuáles columnas fueron pivote
        resultado += f"\nLas columnas pivote son: {', '.join(map(str, columnas_pivote))}.\n"
        return resultado

    # ---------------------------------------
    # Entrada por teclado (flujo interactivo)
    # ---------------------------------------
    @staticmethod
    def leer_matriz_desde_teclado():
        """
        Pide al usuario:
          - número de ecuaciones (m)
          - número de incógnitas (n)
          - m filas con n+1 números (coeficientes y término independiente)
        Devuelve una instancia de SistemaLineal lista para resolver.
        """
        # Leemos m y n con validación básica
        while True:
            try:
                m = int(input("Número de ecuaciones (filas): ").strip())
                n = int(input("Número de incógnitas (columnas): ").strip())
                if m <= 0 or n <= 0:
                    print("Por favor, ingresa valores positivos.")
                    continue
                break
            except ValueError:
                print("Entrada no válida. Intenta de nuevo.")

        print("\nIntroduce cada fila de la MATRIZ AUMENTADA con", n + 1, "valores separados por espacio.")
        print("Ejemplo (n=3):  2 -1 3 5   ← equivale a [2, -1, 3 | 5]\n")

        # Sin usar .append: construimos la matriz con comprensión de listas
        matriz = [
            # Para cada i, pedimos una línea, limpiamos el '|' si lo escribe el usuario,
            # partimos por espacios y convertimos a float.
            (lambda partes: [float(x) for x in partes])(
                input(f"Fila {i+1}: ").replace("|", " ").split()
            )
            for i in range(m)
        ]

        # Validación de longitud por fila
        for i, fila in enumerate(matriz, start=1):
            if len(fila) != n + 1:
                raise ValueError(f"La fila {i} no tiene {n+1} números.")

        return SistemaLineal(matriz)

    @staticmethod
    def resolver_desde_teclado():
        """
        Flujo completo: leer matriz, ejecutar eliminación gaussiana e imprimir pasos y conclusión.
        """
        sistema = SistemaLineal.leer_matriz_desde_teclado()
        print("\n=== Resolviendo por Eliminación Gaussiana (con pasos) ===")
        print(sistema.eliminacion_gaussiana())


# -------------------
# Punto de entrada
# -------------------
if __name__ == "__main__":
    # Ejecuta el flujo interactivo de consola
    SistemaLineal.resolver_desde_teclado()
