# utilidad.py
from __future__ import annotations
from typing import List, Tuple
import ast
import operator as op
from fractions import Fraction

# -------------------------
#  Constantes globales
# -------------------------
DEFAULT_EPS: float = 1e-10
DEFAULT_DEC: int = 4

# -------------------------
#  Evaluador seguro
# -------------------------
# Operadores permitidos
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: lambda x: x
}
# Nodos permitidos
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num)

def _eval_node(node):
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError("Expresión no permitida.")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Constante no numérica.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1e6:
            raise ValueError("Exponente demasiado grande.")
        try:
            return _OPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ValueError("División por cero.")
    raise ValueError("Expresión no permitida.")

def evaluar_expresion(txt: str, exacto: bool = False):
    """
    Evalúa expresiones aritméticas simples: 1/2, -3.5, 2^3.
    Si exacto=True intenta devolver Fraction cuando aplica.
    """
    txt = str(txt).strip()
    if not txt:
        raise ValueError("Vacío")
    txt = txt.replace("^", "**")
    try:
        tree = ast.parse(txt, mode="eval")
    except Exception:
        raise ValueError("Expresión inválida.")
    val = _eval_node(tree.body)
    if exacto:
        try:
            return Fraction(val).limit_denominator()
        except Exception:
            pass
    return float(val)

# =========================
#   Números, vectores, matrices
# =========================
def is_close(a: float, b: float, tol: float = DEFAULT_EPS) -> bool:
    return abs(a - b) < tol

def zeros(m: int, n: int):
    if m <= 0 or n <= 0:
        raise ValueError("Dimensiones inválidas para matriz de ceros.")
    return [[0.0 for _ in range(n)] for _ in range(m)]

def eye(n: int):
    if n <= 0:
        raise ValueError("Dimensión inválida para identidad.")
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def copy_mat(A: List[List[float]]):
    if not A or not A[0]:
        return []
    ancho = len(A[0])
    for i, fila in enumerate(A, 1):
        if len(fila) != ancho:
            raise ValueError(f"Matriz no rectangular (fila {i}).")
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
    if not A or not A[0]:
        return []
    m, n = len(A), len(A[0])
    for i, fila in enumerate(A, 1):
        if len(fila) != n:
            raise ValueError(f"Matriz no rectangular (fila {i}).")
    return [[A[i][j] for i in range(m)] for j in range(n)]

def mat_from_columns(cols: List[List[float]]) -> List[List[float]]:
    """
    Construye una matriz n x k a partir de una lista de k columnas de longitud n.
    """
    if not cols:
        return []
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("Columnas de distinta longitud.")
    k = len(cols)
    return [[cols[j][i] for j in range(k)] for i in range(n)]  # n x k

def dot_mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    if not A or not A[0]:
        return []
    m, n = len(A), len(A[0])
    for i, fila in enumerate(A, 1):
        if len(fila) != n:
            raise ValueError(f"Matriz no rectangular (fila {i}).")
    if len(v) != n:
        raise ValueError("Dimensiones incompatibles en A·v.")
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(m)]
def formatear_matriz(M, decimales=2):
    """
    Devuelve una representación en texto de la matriz M
    con los valores formateados y tabulados.
    """
    if not M:
        return "[ ]"
    lineas = []
    for fila in M:
        fila_str = "  ".join(f"{x:.{decimales}f}" for x in fila)
        lineas.append(f"[ {fila_str} ]")
    return "\n".join(lineas)


# =========================
#   Formato amigable
# =========================
def _fmt_num(x: float, dec: int = DEFAULT_DEC) -> str:
    if abs(x) < 10**(-dec):
        x = 0.0
    try:
        return f"{int(x)}" if float(x).is_integer() else f"{x:.{dec}f}"
    except Exception:
        return str(x)

def format_matrix(A: List[List[float]], dec: int = DEFAULT_DEC, sep: str = " ") -> str:
    return "\n".join(sep.join(_fmt_num(x, dec) for x in fila) for fila in A)

def format_matrix_bracket(A: List[List[float]], dec: int = DEFAULT_DEC) -> str:
    rows = []
    for fila in A:
        rows.append("[ " + "  ".join(_fmt_num(x, dec) for x in fila) + " ]")
    return "\n".join(rows)

def format_vector(v: List[float], dec: int = DEFAULT_DEC, sep: str = " ") -> str:
    return sep.join(_fmt_num(x, dec) for x in v)

def format_col_vector(v: List[float], dec: int = DEFAULT_DEC) -> str:
    return "\n".join(f"[{_fmt_num(x, dec)}]" for x in v)

def fmt_number(x: float, dec: int = DEFAULT_DEC, as_fraction: bool = False) -> str:
    """
    Formatea un número como decimal o como fracción acotando denominador.
    """
    if as_fraction:
        try:
            frac = Fraction(x).limit_denominator()
            return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        except Exception:
            pass
    if abs(x) < 10**(-dec):
        x = 0.0
    try:
        return f"{int(x)}" if float(x).is_integer() else f"{x:.{dec}f}"
    except Exception:
        return str(x)

# =========================
#   Entrada CLI
# =========================
def _split_nums(linea: str):
    linea = (linea or "").replace("|", " ")
    partes = [p for p in linea.replace(",", " ").split() if p]
    if len(partes) < 1:
        raise ValueError("Sin datos")
    vals = [evaluar_expresion(p, exacto=False) for p in partes]
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
