# matrices.py
# -*- coding: utf-8 -*-
"""
Operaciones con matrices y verificación de propiedades de la traspuesta.
Sin NumPy. Pensado para integrarse con tu calculadora por menús.
Todas las rutinas devuelven dicts con 'pasos', 'resultado' (si aplica) y 'conclusion'.
"""

from typing import List, Tuple, Dict, Any
from utilidad import evaluar_expresion

Matriz = List[List[float]]
EPS = 1e-9  # tolerancia para comparaciones

# ---------- Utilidades básicas ----------
def es_rectangular(A: Matriz) -> bool:
    return all(len(f) == len(A[0]) for f in A) if A else False

def dims(A: Matriz) -> Tuple[int, int]:
    return (len(A), len(A[0]) if A else 0)

def mismas_dim(A: Matriz, B: Matriz) -> bool:
    return dims(A) == dims(B)

def compatibles_prod(A: Matriz, B: Matriz) -> bool:
    return dims(A)[1] == dims(B)[0]

def copia(A: Matriz) -> Matriz:
    return [fila[:] for fila in A]

def iguales(A: Matriz, B: Matriz, eps: float = EPS) -> bool:
    if not mismas_dim(A, B):
        return False
    m, n = dims(A)
    for i in range(m):
        for j in range(n):
            if abs(A[i][j] - B[i][j]) > eps:
                return False
    return True

# ---------- Operaciones primitivas ----------
def traspuesta(A: Matriz) -> Matriz:
    m, n = dims(A)
    return [[A[i][j] for i in range(m)] for j in range(n)]

def suma(A: Matriz, B: Matriz) -> Matriz:
    if not mismas_dim(A, B):
        raise ValueError("Para A + B se requieren dimensiones iguales.")
    m, n = dims(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]

def resta(A: Matriz, B: Matriz) -> Matriz:
    if not mismas_dim(A, B):
        raise ValueError("Para A - B se requieren dimensiones iguales.")
    m, n = dims(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]

def escalar_por_matriz(k: float, A: Matriz) -> Matriz:
    m, n = dims(A)
    return [[k * A[i][j] for j in range(n)] for i in range(m)]

def producto(A: Matriz, B: Matriz) -> Matriz:
    if not compatibles_prod(A, B):
        raise ValueError("Para AB, #columnas(A) debe igualar #filas(B).")
    m, p = dims(A)
    p2, n = dims(B)
    assert p == p2
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for k in range(p):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]
    return C

# ---------- Wrappers explicados ----------
def suma_matrices_explicada(A: Matriz, B: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not mismas_dim(A, B):
        return {"pasos": pasos, "conclusion": "No se pueden sumar: dimensiones distintas."}
    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para suma.")
    C = suma(A, B)
    pasos.append("C = A + B calculada elemento a elemento.")
    CT = traspuesta(C)
    AT, BT = traspuesta(A), traspuesta(B)
    suma_Ts = suma(AT, BT)
    cumple = iguales(CT, suma_Ts)
    pasos.append("Se calculó (A + B)^T y A^T + B^T para comparar.")
    conclusion = "Se cumple la propiedad (A + B)^T = A^T + B^T." if cumple else \
                 "No se cumple la propiedad (A + B)^T = A^T + B^T."
    return {"pasos": pasos, "resultado": C, "traspuesta_del_resultado": CT, "AT": AT, "BT": BT,
            "AT_mas_BT": suma_Ts, "conclusion": conclusion}

def resta_matrices_explicada(A: Matriz, B: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not mismas_dim(A, B):
        return {"pasos": pasos, "conclusion": "No se pueden restar: dimensiones distintas."}
    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para resta.")
    C = resta(A, B)
    pasos.append("C = A - B calculada elemento a elemento.")
    CT = traspuesta(C)
    AT, BT = traspuesta(A), traspuesta(B)
    resta_Ts = resta(AT, BT)
    cumple = iguales(CT, resta_Ts)
    pasos.append("Se calculó (A - B)^T y A^T - B^T para comparar.")
    conclusion = "Se cumple la propiedad (A - B)^T = A^T - B^T." if cumple else \
                 "No se cumple la propiedad (A - B)^T = A^T - B^T."
    return {"pasos": pasos, "resultado": C, "traspuesta_del_resultado": CT, "AT": AT, "BT": BT,
            "AT_menos_BT": resta_Ts, "conclusion": conclusion}

def producto_escalar_explicado(k: float, A: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A):
        return {"pasos": pasos, "conclusion": "La matriz A no es rectangular."}
    pasos.append(f"Se multiplicó cada entrada de A por k = {k:g}.")
    C = escalar_por_matriz(k, A)
    CT = traspuesta(C)
    AT = traspuesta(A)
    kAT = escalar_por_matriz(k, AT)
    cumple = iguales(CT, kAT)
    pasos.append("Se comparó (kA)^T con k·A^T.")
    conclusion = "Se cumple la propiedad (kA)^T = kA^T." if cumple else \
                 "No se cumple la propiedad (kA)^T = kA^T."
    return {"pasos": pasos, "resultado": C, "traspuesta_del_resultado": CT, "AT": AT, "kAT": kAT,
            "conclusion": conclusion}

def producto_matrices_explicado(A: Matriz, B: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not compatibles_prod(A, B):
        return {"pasos": pasos, "conclusion": "No se puede multiplicar: columnas(A) ≠ filas(B)."}
    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para producto.")
    C = producto(A, B)
    pasos.append("C = A·B con regla i,j: C[i][j] = sum_k A[i][k]·B[k][j].")
    CT = traspuesta(C)
    AT, BT = traspuesta(A), traspuesta(B)
    BT_AT = producto(BT, AT)
    cumple = iguales(CT, BT_AT)
    pasos.append("Se calculó (AB)^T y B^T·A^T para comparar.")
    conclusion = "Se cumple la propiedad (AB)^T = B^T·A^T." if cumple else \
                 "No se cumple la propiedad (AB)^T = B^T·A^T."
    return {"pasos": pasos, "resultado": C, "traspuesta_del_resultado": CT, "AT": AT, "BT": BT,
            "BT_por_AT": BT_AT, "conclusion": conclusion}

def traspuesta_explicada(A: Matriz) -> Dict[str, Any]:
    pasos = []
    if not es_rectangular(A):
        return {"pasos": pasos, "conclusion": "La matriz A no es rectangular."}
    m, n = dims(A)
    pasos.append(f"Se intercambian filas↔columnas: A es {m}×{n}, A^T será {n}×{m}.")
    AT = traspuesta(A)
    ATT = traspuesta(AT)
    cumple = iguales(ATT, A)
    pasos.append("Verificación: (A^T)^T comparada con A.")
    conclusion = "Se cumple la propiedad (A^T)^T = A." if cumple else \
                 "No se cumple la propiedad (A^T)^T = A."
    return {"pasos": pasos, "resultado": AT, "ATT": ATT, "conclusion": conclusion}

def verificar_propiedades_traspuesta(A: Matriz, B: Matriz, k: float) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    ATT = traspuesta(traspuesta(A))
    cumple_a = iguales(ATT, A)
    info["a"] = {"se_cumple": cumple_a,
                 "mensaje": "Se cumple la propiedad (A^T)^T = A." if cumple_a
                            else "No se cumple la propiedad (A^T)^T = A."}
    if mismas_dim(A, B):
        izq_b = traspuesta(suma(A, B))
        der_b = suma(traspuesta(A), traspuesta(B))
        cumple_b = iguales(izq_b, der_b)
        info["b_suma"] = {"se_cumple": cumple_b,
                          "mensaje": "Se cumple (A+B)^T=A^T+B^T." if cumple_b
                                     else "No se cumple (A+B)^T=A^T+B^T."}
    izq_c = traspuesta(escalar_por_matriz(k, A))
    der_c = escalar_por_matriz(k, traspuesta(A))
    cumple_c = iguales(izq_c, der_c)
    info["c"] = {"se_cumple": cumple_c,
                 "mensaje": "Se cumple (kA)^T=kA^T." if cumple_c
                            else "No se cumple (kA)^T=kA^T."}
    return info

def propiedad_r_suma_traspuesta_explicada(A: Matriz, B: Matriz, r: float) -> Dict[str, Any]:
    pasos: List[str] = []
    if not es_rectangular(A) or not es_rectangular(B):
        return {"pasos": pasos, "conclusion": "Alguna matriz no es rectangular."}
    if not mismas_dim(A, B):
        return {"pasos": pasos, "conclusion": "A y B deben tener las mismas dimensiones."}
    pasos.append(f"Dimensiones: A{dims(A)}, B{dims(B)} → compatibles para suma.")
    pasos.append(f"Escalar r = {r:g}")
    S = suma(A, B)
    pasos.append("S = A + B.")
    rS = escalar_por_matriz(r, S)
    pasos.append("rS = r·(A + B).")
    izq = traspuesta(rS)
    pasos.append("Izquierda: (r(A+B))^T.")
    AT, BT = traspuesta(A), traspuesta(B)
    ATmBT = suma(AT, BT)
    der = escalar_por_matriz(r, ATmBT)
    cumple = iguales(izq, der)
    conclusion = "Se cumple (r(A+B))^T = r(A^T+B^T)." if cumple else \
                 "No se cumple (r(A+B))^T = r(A^T+B^T)."
    return {"pasos": pasos, "S": S, "rS": rS, "izquierda": izq,
            "AT": AT, "BT": BT, "AT_mas_BT": ATmBT,
            "derecha": der, "conclusion": conclusion}

# =========================
# Evaluador de expresiones matriciales
# =========================
class _Tok:
    def __init__(self, kind, value=None):
        self.kind = kind  # 'num' | 'mat' | 'op' | '(' | ')' | 'eof'
        self.value = value

def _tokenize_matrix_expr(s: str) -> list[_Tok]:
    i, n = 0, len(s)
    toks: list[_Tok] = []
    OPS = set('+-*()[]')
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1; continue
        if c in '+-*()':
            toks.append(_Tok('op', c) if c in '+-*' else _Tok(c))
            i += 1; continue
        if c == '[':
            i += 1; start = i; buf = []
            depth = 1
            while i < n and depth > 0:
                if s[i] == '[': depth += 1
                elif s[i] == ']':
                    depth -= 1
                    if depth == 0: break
                buf.append(s[i]); i += 1
            if depth != 0: raise ValueError("Falta ']' de cierre.")
            mat_text = ''.join(buf); i += 1
            toks.append(_Tok('mat', mat_text)); continue
        j = i
        while j < n and s[j] not in OPS and not s[j].isspace():
            j += 1
        atom = s[i:j]; toks.append(_Tok('num', atom)); i = j
    toks.append(_Tok('eof')); return toks

def _parse_matrix_literal(text: str) -> Matriz:
    rows_raw = [ln.strip() for ln in text.replace(';', '\n').splitlines() if ln.strip()]
    rows: Matriz = []
    width = None
    for ln in rows_raw:
        parts = [p for p in ln.replace(',', ' ').split() if p]
        vals = [float(evaluar_expresion(p)) for p in parts]
        if width is None: width = len(vals)
        elif len(vals) != width: raise ValueError("Matriz no rectangular.")
        rows.append(vals)
    return rows

def _to_value(tok: _Tok):
    if tok.kind == 'num':
        return float(evaluar_expresion(tok.value)), 'scalar'
    if tok.kind == 'mat':
        return _parse_matrix_literal(tok.value), 'matrix'
    raise ValueError("Token inesperado.")

class _Parser:
    def __init__(self, toks: list[_Tok]):
        self.toks = toks; self.k = 0; self.pasos = []

    def _cur(self): return self.toks[self.k]
    def _eat(self, kind=None, val=None):
        t = self._cur()
        self.k += 1; return t

    def parse(self):
        val, typ = self._expr()
        return val, typ, self.pasos

    def _expr(self):
        left, tleft = self._term()
        while self._cur().kind == 'op' and self._cur().value in '+-':
            op = self._eat('op').value
            right, tright = self._term()
            if tleft == 'matrix' and tright == 'matrix':
                left = suma(left, right) if op == '+' else resta(left, right)
                self.pasos.append(f"Aplicar {op} entre matrices.")
                tleft = 'matrix'
            elif tleft == 'scalar' and tright == 'scalar':
                left = left + right if op == '+' else left - right
                self.pasos.append("Operación aritmética de escalares.")
            else:
                raise ValueError("No se puede sumar/restar escalar con matriz.")
        return left, tleft

    def _term(self):
        left, tleft = self._factor()
        while self._cur().kind == 'op' and self._cur().value == '*':
            self._eat('op')
            right, tright = self._factor()
            if tleft == 'scalar' and tright == 'matrix':
                left = escalar_por_matriz(left, right)
                self.pasos.append("Escalar por matriz (izquierda)."); tleft = 'matrix'
            elif tleft == 'matrix' and tright == 'scalar':
                left = escalar_por_matriz(right, left)
                self.pasos.append("Escalar por matriz (derecha)."); tleft = 'matrix'
            elif tleft == 'matrix' and tright == 'matrix':
                left = producto(left, right)
                self.pasos.append("Producto de matrices."); tleft = 'matrix'
            elif tleft == 'scalar' and tright == 'scalar':
                left *= right; self.pasos.append("Producto escalar."); tleft = 'scalar'
            else:
                raise ValueError("Producto no válido.")
        return left, tleft

    def _factor(self):
        t = self._cur()
        if t.kind == '(':
            self._eat('('); val, typ = self._expr(); self._eat(')'); return val, typ
        if t.kind in ('num', 'mat'):
            self._eat(t.kind); return _to_value(t)
        raise ValueError("Factor inválido.")

def evaluar_expresion_matricial(texto: str) -> Dict[str, Any]:
    toks = _tokenize_matrix_expr(texto)
    parser = _Parser(toks)
    val, typ, pasos = parser.parse()
    return {"resultado": val, "tipo": typ, "pasos": pasos}
