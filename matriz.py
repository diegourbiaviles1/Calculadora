def eliminacion_gaussiana(A, b):
    n = len(A)

    # Convertimos a matriz aumentada
    for i in range(n):
        A[i].append(b[i])

    # Proceso de eliminación
    for i in range(n):
        # Pivoteo: asegurarnos que A[i][i] no es 0
        if A[i][i] == 0:
            for j in range(i + 1, n):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    break

        # Normalizar fila pivote
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("El sistema no tiene solución única.")
        for k in range(i, n + 1):
            A[i][k] /= pivot

        # Eliminar debajo del pivote
        for j in range(i + 1, n):
            factor = A[j][i]
            for k in range(i, n + 1):
                A[j][k] -= factor * A[i][k]

    # Sustitución hacia atrás
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))
    return x


def main():
    print("=== Resolución de sistemas con Eliminación Gaussiana ===")
    n = int(input("Ingrese el tamaño de la matriz (2x2 o 3x3): "))

    if n not in [2, 3]:
        print("Solo se permiten matrices 2x2 o 3x3.")
        return

    A = []
    b = []

    print(f"Ingrese los coeficientes de la matriz {n}x{n}:")
    for i in range(n):
        fila = []
        for j in range(n):
            val = float(input(f"A[{i+1}][{j+1}]: "))
            fila.append(val)
        A.append(fila)

    print("Ingrese los valores del vector de resultados:")
    for i in range(n):
        val = float(input(f"b[{i+1}]: "))
        b.append(val)

    try:
        solucion = eliminacion_gaussiana(A, b)
        print("\n--- Solución del sistema ---")
        for i, val in enumerate(solucion):
            print(f"x{i+1} = {val:.4f}")
    except ValueError as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
