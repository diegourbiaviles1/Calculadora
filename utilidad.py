# utilidad.py
from __future__ import annotations
from typing import List, Tuple
import ast
import operator as op
from fractions import Fraction
import math  # <--- AGREGADO: Necesario para funciones matemáticas

# -------------------------
#  Constantes globales
# -------------------------
DEFAULT_EPS: float = 1e-10
DEFAULT_DEC: int = 2

# -------------------------
#  Evaluador seguro (MEJORADO)
# -------------------------
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: lambda x: x
}
# Funciones permitidas
_FUNCS = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 
    'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt, 'abs': abs
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, 
    ast.Num, ast.Call, ast.Name, ast.Load
)

def _eval_node(node):
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError("Expresión no permitida.")
    
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)): return float(node.value)
        raise ValueError("Constante no numérica.")
    
    if isinstance(node, ast.Num): return float(node.n)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponente demasiado grande.")
        try: return _OPS[type(node.op)](left, right)
        except ZeroDivisionError: raise ValueError("División por cero.")

    # Soporte para funciones (sin, cos, etc.)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            args = [_eval_node(arg) for arg in node.args]
            return _FUNCS[node.func.id](*args)
        raise ValueError(f"Función no permitida: {node.func.id}")
    
    # Soporte para constantes (pi, e)
    if isinstance(node, ast.Name):
        if node.id == 'pi': return math.pi
        if node.id == 'e': return math.e
        raise ValueError(f"Variable desconocida: {node.id}")

    raise ValueError("Estructura no soportada.")

def evaluar_expresion(txt: str, exacto: bool = False):
    txt = str(txt).strip()
    if not txt: raise ValueError("Vacío")
    txt = txt.replace("^", "**")
    try:
        tree = ast.parse(txt, mode="eval")
    except: raise ValueError("Expresión inválida.")
    
    val = _eval_node(tree.body)
    
    if exacto:
        try: return Fraction(val).limit_denominator()
        except: pass
    return float(val)

# --- FUNCIONES ORIGINALES (Se mantienen igual) ---
def is_close(a, b, tol=DEFAULT_EPS): return abs(a - b) < tol
def zeros(m, n): return [[0.0]*n for _ in range(m)]
def eye(n): 
    I = zeros(n, n)
    for i in range(n): I[i][i] = 1.0
    return I
def copy_mat(A): return [row[:] for row in A] if A else []
def vec_suma(a, b): return [x+y for x,y in zip(a,b)]
def escalar_por_vector(k, v): return [k*x for x in v]
def sumar_vec(u, v): return vec_suma(u, v)
def columnas(A): return [[row[i] for row in A] for i in range(len(A[0]))] if A else []
def mat_from_columns(cols): return [[col[i] for col in cols] for i in range(len(cols[0]))] if cols else []
def dot_mat_vec(A, v): return [sum(A[i][j]*v[j] for j in range(len(v))) for i in range(len(A))]
def formatear_matriz(M, dec=2): return "\n".join([" ".join(f"{x:.{dec}f}" for x in row) for row in M]) if M else "[]"

# Formato
def _fmt_num(x, dec=DEFAULT_DEC): return f"{int(x)}" if float(x).is_integer() else f"{x:.{dec}f}"
def format_matrix(A, dec=DEFAULT_DEC, sep=" "): return "\n".join(sep.join(_fmt_num(x, dec) for x in r) for r in A)
def format_matrix_bracket(A, dec=DEFAULT_DEC): return "\n".join(f"[ {' '.join(_fmt_num(x, dec) for x in r)} ]" for r in A)
def format_vector(v, dec=DEFAULT_DEC, sep=" "): return sep.join(_fmt_num(x, dec) for x in v)
def format_col_vector(v, dec=DEFAULT_DEC): return "\n".join(f"[{_fmt_num(x, dec)}]" for x in v)
def fmt_number(x, dec=DEFAULT_DEC, as_fraction=False): return _fmt_num(x, dec)

# CLI
def _split_nums(s): return [evaluar_expresion(p, False) for p in s.replace("|", " ").replace(",", " ").split()]
def leer_vector(n, msg="Vector: "): return _split_nums(input(msg))
def leer_matriz(m, n, t=""): return [leer_vector(n) for _ in range(m)]
def leer_lista_vectores(k, n, t=""): return [leer_vector(n) for _ in range(k)]
def leer_dimensiones(msg=""): return int(input().split()[0]), int(input().split()[1])