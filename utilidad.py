# ---------- Lectura desde teclado ----------
import ast
import operator as op

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}

def _eval_node(node):
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("Expresión no permitida.")

def evaluar_expresion(txt: str) -> float:
    return _eval_node(ast.parse(str(txt), mode='eval').body)

def is_close(a, b, tol=1e-10):
    return abs(a - b) < tol

def zeros(m, n):
    return [[0.0 for _ in range(n)] for _ in range(m)]

def vec_suma(*vs):
    if not vs: return []
    n = len(vs[0])
    r = [0.0]*n
    for v in vs:
        if len(v)!=n: raise ValueError("Vectores de distinta longitud")
        for i in range(n): r[i]+=v[i]
    return r

def escalar_por_vector(a, v):
    return [a*vi for vi in v]

def col_matrix(vectores):
    if not vectores: return []
    n = len(vectores[0])
    for v in vectores:
        if len(v)!=n: raise ValueError("Todos los vectores deben tener la misma longitud")
    k = len(vectores)
    A = zeros(n, k)
    for j, v in enumerate(vectores):
        for i in range(n):
            A[i][j] = float(v[i])
    return A

def _split_nums(linea: str):
    if "|" in linea:
        linea = linea.replace("|", " ")
    partes = [p for p in linea.replace(",", " ").split() if p]
    return [evaluar_expresion(p) for p in partes]

def leer_vector(n=None, prompt="Vector: "):
    while True:
        try:
            linea = input(prompt)
            v = _split_nums(linea)
            if n is not None and len(v) != n:
                print(f"Se esperaban {n} componentes. Intenta de nuevo.")
                continue
            return v
        except Exception as e:
            print("Entrada inválida:", e)

def leer_lista_vectores(k, n, titulo="Introduce los vectores (cada línea separada por comas/espacios):"):
    print(titulo)
    vectores = []
    for idx in range(1, k+1):
        v = leer_vector(n, f"v{idx}: ")
        vectores.append(v)
    return vectores

def leer_matriz(m, n, titulo="Introduce las filas (separadas por espacios/comas):"):
    print(titulo)
    A = []
    for i in range(1, m+1):
        fila = leer_vector(n, f"Fila {i}: ")
        A.append(fila)
    return A

def leer_dimensiones(msg="Dimensiones m n: "):
    while True:
        try:
            m, n = _split_nums(input(msg))
            m, n = int(m), int(n)
            if m<=0 or n<=0: raise ValueError
            return m, n
        except Exception:
            print("Ingresa dos enteros positivos (ej: 3 2).")
