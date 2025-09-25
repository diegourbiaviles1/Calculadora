
from typing import List, Tuple
import ast
import operator as op

_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg
}

def _eval_node(node):
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("Expresión no permitida.")

def evaluar_expresion(txt: str) -> float:
    """Evalúa expresiones aritméticas simples: 1/2, -3.5, 2^3, etc."""
    txt = str(txt).strip()
    if not txt:
        raise ValueError("Vacío")
    # permitir ^ como potencia
    txt = txt.replace("^", "**")
    return _eval_node(ast.parse(txt, mode="eval").body)

# =========================
#   Números, vectores, matrices
# =========================
def is_close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) < tol

def zeros(m: int, n: int):
    return [[0.0 for _ in range(n)] for _ in range(m)]

def eye(n: int):
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def copy_mat(A: List[List[float]]):
    return [fila[:] for fila in A]

def vec_suma(a: List[float], b: List[float]) -> List[float]:
    if len(a) != len(b):
        raise ValueError("Dimensiones incompatibles en suma de vectores.")
    return [ai + bi for ai, bi in zip(a, b)]

def escalar_por_vector(k: float, v: List[float]) -> List[float]:
    return [k * x for x in v]

def sumar_vec(u: List[float], v: List[float]) -> List[float]:
    return vec_suma(u, v)

def columnas(A: List[List[float]]) -> List[List[float]]:
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def dot_mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    m, n = len(A), len(A[0])
    if len(v) != n:
        raise ValueError("Dimensiones incompatibles en A·v.")
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(m)]

# =========================
#   Formato amigable
# =========================
def _fmt_num(x: float, dec: int = 4) -> str:
    # limpia ceros muy pequeños
    if abs(x) < 10**(-dec):
        x = 0.0
    return f"{int(x)}" if float(x).is_integer() else f"{x:.{dec}f}"

def format_matrix(A: List[List[float]], dec: int = 4, sep: str = " ") -> str:
    return "\n".join(sep.join(_fmt_num(x, dec) for x in fila) for fila in A)

def format_matrix_bracket(A: List[List[float]], dec: int = 4) -> str:
    rows = []
    for fila in A:
        rows.append("[ " + "  ".join(_fmt_num(x, dec) for x in fila) + " ]")
    return "\n".join(rows)

def format_vector(v: List[float], dec: int = 4, sep: str = " ") -> str:
    return sep.join(_fmt_num(x, dec) for x in v)

def format_col_vector(v: List[float], dec: int = 4) -> str:
    return "\n".join(f"[{_fmt_num(x, dec)}]" for x in v)

# =========================
#   Entrada CLI
# =========================
def _split_nums(linea: str):
    linea = linea.replace("|", " ")
    partes = [p for p in linea.replace(",", " ").split() if p]
    if len(partes) < 1:
        raise ValueError("Sin datos")
    vals = [evaluar_expresion(p) for p in partes]
    return vals

def leer_vector(n: int, msg: str = "Vector: ") -> List[float]:
    while True:
        try:
            vals = _split_nums(input(msg))
            if len(vals) != n:
                raise ValueError(f"Se esperaban {n} valores")
            return [float(x) for x in vals]
        except Exception as e:
            print("Entrada inválida:", e)

def leer_matriz(m: int, n: int, titulo="Introduce las filas (separadas por espacios/comas):") -> List[List[float]]:
    print(titulo)
    A = []
    for i in range(1, m + 1):
        A.append(leer_vector(n, f"Fila {i}: "))
    return A

def leer_lista_vectores(k: int, n: int, titulo="Introduce los vectores (una línea por vector):") -> List[List[float]]:
    print(titulo)
    V = []
    for i in range(1, k + 1):
        V.append(leer_vector(n, f"v{i}: "))
    return V

def leer_dimensiones(msg="Dimensiones m n: ") -> Tuple[int, int]:
    while True:
        try:
            m, n = _split_nums(input(msg))
            m, n = int(m), int(n)
            if m <= 0 or n <= 0:
                raise ValueError
            return m, n
        except Exception:
            print("Ingresa dos enteros positivos (ej: 3 2).")
