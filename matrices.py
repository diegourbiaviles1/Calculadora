# Reemplaza/ajusta en matrices.py

from utilidad import DEFAULT_EPS
# === helpers necesarios ===
def _transpose(M):
    """Devuelve la traspuesta de M."""
    if not M or not M[0]:
        return []
    m, n = len(M), len(M[0])
    # columnas j se vuelven filas
    return [[float(M[i][j]) for i in range(m)] for j in range(n)]

def _eq_same_size(X, Y, tol=DEFAULT_EPS):
    """Compara X e Y (mismo tamaño) con tolerancia numérica."""
    if not X or not Y or not X[0] or not Y[0]:
        return False
    if len(X) != len(Y) or len(X[0]) != len(Y[0]):
        return False
    m, n = len(X), len(X[0])
    for i in range(m):
        for j in range(n):
            if abs(float(X[i][j]) - float(Y[i][j])) > tol:
                return False
    return True


def _format_rows_2dec(M):
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

    pasos.append(f"Paso 1 (Matrices de entrada):\nA:\n{_format_rows_2dec(A)}\nB:\n{_format_rows_2dec(B)}\n")

    # Suma normal C = A + B
    C = [[0.0]*n for _ in range(m)]
    pasos.append("Paso 2 (Cálculo elemento a elemento de C=A+B):")
    for i in range(m):
        for j in range(n):
            a_ij = float(A[i][j]); b_ij = float(B[i][j])
            C[i][j] = a_ij + b_ij
            pasos.append(f"  c{i+1}{j+1} = {a_ij:.{dec}f} + {b_ij:.{dec}f} = {C[i][j]:.{dec}f}")

    pasos.append(f"\nPaso 3 (Resultado C=A+B):\n{_format_rows_2dec(C)}\n")

    # Verificación (A+B)^T = A^T + B^T
    CT = _transpose(C)
    AT = _transpose(A)
    BT = _transpose(B)
    S = [[AT[i][j] + BT[i][j] for j in range(len(AT[0]))] for i in range(len(AT))]

    ver = []
    ver.append("Verificación de propiedad: (A + B)^T = A^T + B^T")
    ver.append("  (A+B)^T =")
    ver.append(_format_rows_2dec(CT))
    ver.append("  A^T + B^T =")
    ver.append(_format_rows_2dec(S))
    equivalentes = _eq_same_size(CT, S)
    ver.append("  ¿Son iguales? → " + ("Sí, se cumple" if equivalentes else "No, no se cumple"))

    return {
        "ok": True,
        "C": C,
        "pasos": pasos,
        "mensaje": "Suma realizada correctamente.",
        "propiedad": "(A + B)^T = A^T + B^T",
        "equivalentes": equivalentes,
        "detalle_verificacion": "\n".join(ver),
    }


def resta_matrices_explicada(A, B, dec: int = 2):
    if not A or not B or not A[0] or not B[0]:
        return {"ok": False, "mensaje": "Matrices vacías o inválidas.", "pasos": []}
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return {"ok": False, "mensaje": "No se puede restar: dimensiones distintas.", "pasos": []}

    m, n = len(A), len(A[0])
    pasos = []
    pasos.append(f"Paso 1 (Matrices de entrada):\nA:\n{_format_rows_2dec(A)}\nB:\n{_format_rows_2dec(B)}\n")

    # Resta normal D = A − B
    D = [[0.0]*n for _ in range(m)]
    pasos.append("Paso 2 (Cálculo elemento a elemento de D=A−B):")
    for i in range(m):
        for j in range(n):
            a_ij = float(A[i][j]); b_ij = float(B[i][j])
            D[i][j] = a_ij - b_ij
            pasos.append(f"  d{i+1}{j+1} = {a_ij:.{dec}f} - {b_ij:.{dec}f} = {D[i][j]:.{dec}f}")

    pasos.append(f"\nPaso 3 (Resultado D=A−B):\n{_format_rows_2dec(D)}\n")

    # Verificación (A−B)^T = A^T − B^T
    DT = _transpose(D)
    AT = _transpose(A)
    BT = _transpose(B)
    R = [[AT[i][j] - BT[i][j] for j in range(len(AT[0]))] for i in range(len(AT))]

    ver = []
    ver.append("Verificación de propiedad: (A − B)^T = A^T − B^T")
    ver.append("  (A−B)^T =")
    ver.append(_format_rows_2dec(DT))
    ver.append("  A^T − B^T =")
    ver.append(_format_rows_2dec(R))
    equivalentes = _eq_same_size(DT, R)
    ver.append("  ¿Son iguales? → " + ("Sí, se cumple" if equivalentes else "No, no se cumple"))

    return {
        "ok": True,
        "C": D,
        "pasos": pasos,
        "mensaje": "Resta realizada correctamente.",
        "propiedad": "(A − B)^T = A^T − B^T",
        "equivalentes": equivalentes,
        "detalle_verificacion": "\n".join(ver),
    }


def escalar_por_matriz_explicada(k, A, dec: int = 2):
    if not A or not A[0]:
        return {"ok": False, "mensaje": "Matriz vacía o inválida.", "pasos": []}
    k = float(k)

    m, n = len(A), len(A[0])
    pasos = []
    pasos.append(f"Paso 1 (Matriz de entrada):\nA:\n{_format_rows_2dec(A)}\n")

    # kA
    B = [[0.0]*n for _ in range(m)]
    pasos.append(f"Paso 2 (Cálculo elemento a elemento): B = ({k:.{dec}f}) · A")
    for i in range(m):
        for j in range(n):
            a_ij = float(A[i][j])
            B[i][j] = k * a_ij
            pasos.append(f"  b{i+1}{j+1} = {k:.{dec}f} · {a_ij:.{dec}f} = {B[i][j]:.{dec}f}")

    pasos.append(f"\nPaso 3 (Resultado B=kA):\n{_format_rows_2dec(B)}\n")

    # Verificación (kA)^T = k A^T
    BT = _transpose(B)
    AT = _transpose(A)
    kAT = [[k * AT[i][j] for j in range(len(AT[0]))] for i in range(len(AT))]

    ver = []
    ver.append("Verificación de propiedad: (kA)^T = k A^T")
    ver.append("  (kA)^T =")
    ver.append(_format_rows_2dec(BT))
    ver.append("  k·A^T =")
    ver.append(_format_rows_2dec(kAT))
    equivalentes = _eq_same_size(BT, kAT)
    ver.append("  ¿Son iguales? → " + ("Sí, se cumple" if equivalentes else "No, no se cumple"))

    return {
        "ok": True,
        "B": B,
        "pasos": pasos,
        "mensaje": "Multiplicación por escalar realizada correctamente.",
        "propiedad": "(kA)^T = k A^T",
        "equivalentes": equivalentes,
        "detalle_verificacion": "\n".join(ver),
    }


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

