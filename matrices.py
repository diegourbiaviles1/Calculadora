# Reemplaza/ajusta en matrices.py

from utilidad import DEFAULT_EPS

def _format_rows_2dec(M):
    # Formato “CalcX”: filas con números a 2 decimales, sin corchetes
    if not M: 
        return ""
    out = []
    for fila in M:
        out.append("  ".join(f"{float(x):.2f}" for x in fila))
    return "\n".join(out)

def suma_matrices_explicada(A, B, dec: int = 2):
    if not A or not B or not A[0] or not B[0]:
        return {"ok": False, "mensaje": "Matrices vacías o inválidas.", "pasos": []}
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return {"ok": False, "mensaje": "No se puede sumar: dimensiones distintas.", "pasos": []}

    m, n = len(A), len(A[0])
    pasos = []

    # Paso 1: mostrar matrices
    pasos.append(f"Paso 1 (Matrices de entrada):\nA:\n{_format_rows_2dec(A)}\nB:\n{_format_rows_2dec(B)}\n")

    # Paso 2: cálculo elemento a elemento con detalle
    C = [[0.0]*n for _ in range(m)]
    pasos.append("Paso 2 (Cálculo elemento a elemento):")
    for i in range(m):
        for j in range(n):
            a_ij = float(A[i][j])
            b_ij = float(B[i][j])
            C[i][j] = a_ij + b_ij
            pasos.append(f"  c{i+1}{j+1} = {a_ij:.2f} + {b_ij:.2f} = {C[i][j]:.2f}")

    # Paso 3: resultado final (matriz completa)
    pasos.append(f"\nPaso 3 (Resultado):\n{_format_rows_2dec(C)}\n")

    return {
        "ok": True,
        "C": C,
        "pasos": pasos,
        "mensaje": "Suma realizada correctamente."
    }


def resta_matrices_explicada(A, B, dec: int = 2):
    if not A or not B or not A[0] or not B[0]:
        return {"ok": False, "mensaje": "Matrices vacías o inválidas.", "pasos": []}
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return {"ok": False, "mensaje": "No se puede restar: dimensiones distintas.", "pasos": []}

    m, n = len(A), len(A[0])
    pasos = []
    pasos.append(f"Paso 1 (Matrices de entrada):\nA:\n{_format_rows_2dec(A)}\nB:\n{_format_rows_2dec(B)}\n")

    C = [[0.0]*n for _ in range(m)]
    pasos.append("Paso 2 (Cálculo elemento a elemento):")
    for i in range(m):
        for j in range(n):
            a_ij = float(A[i][j])
            b_ij = float(B[i][j])
            C[i][j] = a_ij - b_ij
            pasos.append(f"  c{i+1}{j+1} = {a_ij:.{dec}f} - {b_ij:.{dec}f} = {C[i][j]:.{dec}f}")

    pasos.append(f"\nPaso 3 (Resultado):\n{_format_rows_2dec(C)}\n")
    return {"ok": True, "C": C, "pasos": pasos, "mensaje": "Resta realizada correctamente."}

def escalar_por_matriz_explicada(k, A, dec: int = 2):
    if not A or not A[0]:
        return {"ok": False, "mensaje": "Matriz vacía o inválida.", "pasos": []}
    k = float(k)

    m, n = len(A), len(A[0])
    pasos = []
    pasos.append(f"Paso 1 (Matriz de entrada):\nA:\n{_format_rows_2dec(A)}\n")

    B = [[0.0]*n for _ in range(m)]
    pasos.append(f"Paso 2 (Cálculo elemento a elemento): B = ({k:.{dec}f}) · A")
    for i in range(m):
        for j in range(n):
            a_ij = float(A[i][j])
            B[i][j] = k * a_ij
            pasos.append(f"  b{i+1}{j+1} = {k:.{dec}f} · {a_ij:.{dec}f} = {B[i][j]:.{dec}f}")

    pasos.append(f"\nPaso 3 (Resultado):\n{_format_rows_2dec(B)}\n")
    return {"ok": True, "B": B, "pasos": pasos, "mensaje": "Multiplicación por escalar realizada correctamente."}

def multiplicacion_matrices_explicada(A, B, dec: int = 2):
    if not A or not B or not A[0] or not B[0]:
        return {"ok": False, "mensaje": "Matrices vacías o inválidas.", "pasos": []}

    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])

    pasos = []
    pasos.append(f"Paso 1 (Matrices de entrada):\nA:\n{_format_rows_2dec(A)}\nB:\n{_format_rows_2dec(B)}\n")

    if n != n2:
        return {"ok": False, "mensaje": "No se puede multiplicar: columnas(A) ≠ filas(B).", "pasos": pasos}

    C = [[0.0]*p for _ in range(m)]
    pasos.append("Paso 2 (Producto fila×columna, elemento a elemento):")
    for i in range(m):
        for j in range(p):
            terminos = []
            s = 0.0
            for k in range(n):
                a = float(A[i][k])
                b = float(B[k][j])
                terminos.append(f"{a:.{dec}f}·{b:.{dec}f}")
                s += a * b
            C[i][j] = s
            pasos.append(f"  c{i+1}{j+1} = " + " + ".join(terminos) + f" = {s:.{dec}f}")

    pasos.append(f"\nPaso 3 (Resultado):\n{_format_rows_2dec(C)}\n")
    return {"ok": True, "C": C, "pasos": pasos, "mensaje": "Producto AB realizado correctamente."}


def traspuesta_explicada(A, B=None, k=None, dec: int = 2):
    """
    Calcula la traspuesta de A, muestra paso a paso el intercambio de filas por columnas,
    verifica las propiedades básicas y genera mensajes interpretativos claros.
    """
    if not A or not A[0]:
        return {"ok": False, "mensaje": "Matriz vacía o inválida.", "pasos": [], "propiedades": [], "interpretacion": ""}

    m, n = len(A), len(A[0])
    pasos, props = [], []

    def fmt(x): return f"{float(x):.{dec}f}"
    def fmtM(M): return "\n".join("  ".join(fmt(x) for x in fila) for fila in M)

    # Paso 1: Mostrar matriz original
    pasos.append("Paso 1: Matriz original A")
    pasos.append(fmtM(A))

    # Paso 2: Mostrar cómo se obtiene la traspuesta
    pasos.append("\nPaso 2: Intercambio de filas por columnas (A^T)")
    T = [[float(A[i][j]) for i in range(m)] for j in range(n)]
    for i in range(m):
        for j in range(n):
            pasos.append(f"  Elemento a{i+1}{j+1} pasa a posición t{j+1}{i+1}")
    pasos.append("\nResultado A^T:")
    pasos.append(fmtM(T))

    # Propiedades verificadas
    pasos.append("\nPaso 3: Verificación de propiedades básicas")

    # (A^T)^T = A
    TT = [[float(T[i][j]) for i in range(len(T))] for j in range(len(T[0]))]
    cumple1 = all(abs(TT[i][j] - float(A[i][j])) < DEFAULT_EPS for i in range(m) for j in range(n))
    props.append(f"(A^T)^T = A : {'Se cumple' if cumple1 else 'No se cumple'}")

    # (A + B)^T = A^T + B^T
    if B is not None and B and B[0]:
        if len(B) == m and len(B[0]) == n:
            BT = [[float(B[i][j]) for i in range(len(B))] for j in range(len(B[0]))]
            S = [[float(A[i][j]) + float(B[i][j]) for j in range(n)] for i in range(m)]
            lhs = [[float(S[i][j]) for i in range(m)] for j in range(n)]  # (A+B)^T
            rhs = [[T[i][j] + BT[i][j] for j in range(len(T[0]))] for i in range(len(T))]  # A^T+B^T
            cumple2 = all(abs(lhs[i][j] - rhs[i][j]) < DEFAULT_EPS for i in range(len(lhs)) for j in range(len(lhs[0])))
            props.append(f"(A + B)^T = A^T + B^T : {'Se cumple' if cumple2 else 'No se cumple'}")
        else:
            props.append("(A + B)^T = A^T + B^T : no aplica (dimensiones incompatibles).")

    # (kA)^T = kA^T
    if k is not None:
        kf = float(k)
        kA = [[kf * float(A[i][j]) for j in range(n)] for i in range(m)]
        lhs = [[float(kA[i][j]) for i in range(m)] for j in range(n)]  # (kA)^T
        rhs = [[kf * T[i][j] for j in range(len(T[0]))] for i in range(len(T))]  # kA^T
        cumple3 = all(abs(lhs[i][j] - rhs[i][j]) < DEFAULT_EPS for i in range(len(lhs)) for j in range(len(lhs[0])))
        props.append(f"(kA)^T = kA^T : {'Se cumple' if cumple3 else 'No se cumple'}")

    # (AB)^T = B^T A^T
    if B is not None and B and B[0] and len(A[0]) == len(B):
        AB = [[sum(float(A[i][k]) * float(B[k][j]) for k in range(len(B))) for j in range(len(B[0]))] for i in range(m)]
        lhs = [[float(AB[i][j]) for i in range(len(AB))] for j in range(len(AB[0]))]  # (AB)^T
        BT = [[float(B[i][j]) for i in range(len(B))] for j in range(len(B[0]))]
        rhs = [[sum(BT[i][k] * T[k][j] for k in range(len(T))) for j in range(len(T[0]))] for i in range(len(BT))]
        cumple4 = all(abs(lhs[i][j] - rhs[i][j]) < DEFAULT_EPS for i in range(len(lhs)) for j in range(len(lhs[0])))
        props.append(f"(AB)^T = B^T A^T : {'Se cumple' if cumple4 else 'No se cumple'}")

    # Interpretación final
    interpretacion = []
    interpretacion.append("\nInterpretación y Verificación:")
    interpretacion.append("La matriz traspuesta A^T se obtuvo intercambiando filas por columnas.")
    interpretacion.append("Las propiedades verificadas indican las igualdades fundamentales de la traspuesta.")
    interpretacion.append("Por ejemplo, (A^T)^T = A confirma que al trasponer dos veces se recupera la matriz original.")
    interpretacion.append("Cada comparación numérica se realizó elemento a elemento con precisión de tolerancia numérica.\n")

    return {
        "ok": True,
        "A_T": T,
        "pasos": pasos,
        "propiedades": props,
        "mensaje": "Traspuesta calculada y propiedades verificadas correctamente.",
        "interpretacion": "\n".join(interpretacion)
    }

