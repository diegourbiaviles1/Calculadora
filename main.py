# main.py (menú compacto)

from utilidad import leer_matriz, leer_vector, leer_lista_vectores, leer_dimensiones
from algebra_vector import (
    verificar_propiedades, combinacion_lineal, ecuacion_vectorial, resolver_AX_igual_B,
    verificar_distributiva_matriz, sistema_a_forma_matricial, multiplicacion_matriz_vector_explicada
)
from sistema_lineal import SistemaLineal

# -------- Handlers existentes (re-uso) --------
def op0_resolver_sistema():
    print("\n--- Resolver sistema lineal (Gauss-Jordan con pasos) ---")
    m, n = leer_dimensiones("Número de ecuaciones m y número de variables n: ")
    print(f"\nIntroduce la MATRIZ AUMENTADA con {n+1} valores por fila (coeficientes y término independiente).")
    Ab = [leer_vector(n+1, f"Fila {i} (a1 ... an | b): ") for i in range(1, m+1)]
    sl = SistemaLineal(Ab, decimales=4)
    out = sl.gauss_jordan()
    print("\n=== Pasos ===")
    for paso in out["pasos"]: print(paso, "\n")
    if out["tipo"] == "unica":
        print("Solución única x =", out["x"])
    elif out["tipo"] == "infinitas":
        print("Infinitas soluciones. Variables libres en columnas:", out["libres"])
    else:
        print("Sistema inconsistente (sin solución).")

def op1_propiedades():
    print("\n--- Propiedades en R^n ---")
    n = int(input("Dimensión n: "))
    v = leer_vector(n, "v: ")
    u = leer_vector(n, "u: ")
    w = leer_vector(n, "w: ")
    a = float(input("Escalar a: "))
    b = float(input("Escalar b: "))
    res = verificar_propiedades(v,u,w,a,b)
    for k, val in res.items(): print(f"{k}: {val}")

def op2_combinacion():
    print("\n--- Combinación lineal ---")
    k = int(input("Cantidad de vectores k: "))
    n = int(input("Dimensión n: "))
    V = leer_lista_vectores(k, n)
    coef = leer_vector(k, "Coeficientes (c1..ck): ")
    r = combinacion_lineal(V, coef)
    print("Resultado:", r)

def op3_ecuacion_vectorial():
    print("\n--- Ecuación vectorial (¿b en span{v1..vk}?) ---")
    k = int(input("Cantidad de vectores k: "))
    n = int(input("Dimensión n: "))
    V = leer_lista_vectores(k, n)
    b = leer_vector(n, "b: ")
    out = ecuacion_vectorial(V, b)
    for paso in out.get("reportes", []): print(paso, "\n")
    tipo = out.get("tipo") or out.get("estado")
    print("Estado:", tipo)

def op4_ax_igual_b():
    print("\n--- Ecuación matricial AX=B ---")
    m, n = leer_dimensiones("Tamaño de A (m n): ")
    A = leer_matriz(m, n, "Matriz A:")
    es_matriz_B = input("¿B tiene varias columnas? (s/n): ").strip().lower() == "s"
    if es_matriz_B:
        p = int(input("Número de columnas de B (p): "))
        B = leer_matriz(m, p, "Matriz B:")
    else:
        B = leer_vector(m, "Vector b: ")
    out = resolver_AX_igual_B(A, B)
    for paso in out.get("reportes", []): print(paso, "\n")
    if out.get("estado") == "ok":
        if "x" in out: print("x =", out["x"])
        if "X" in out: print("X =\n", out["X"])
    else:
        print("Estado:", out.get("estado"))

def op5_distributiva():
    print("\n--- Verificar distributiva A(u+v) = Au + Av ---")
    m = int(input("Filas de A (m): "))
    n = int(input("Columnas de A (n): "))
    A = leer_matriz(m, n, "Matriz A:")
    u = leer_vector(n, "u (vector columna de tamaño n): ")
    v = leer_vector(n, "v (vector columna de tamaño n): ")
    out = verificar_distributiva_matriz(A, u, v)
    for p in out["pasos"]:
        print(p)
    print("\n" + out["conclusion"])

def op6_sistema_a_matriz():
    print("\n--- Sistema → forma matricial Ax = b ---")
    m = int(input("Número de ecuaciones (m): "))
    n = int(input("Número de variables (n): "))
    print("Introduce la matriz de coeficientes A (m x n):")
    A = leer_matriz(m, n)
    print("Introduce el vector de términos independientes b (m):")
    b = leer_vector(m, "b: ")
    nombres = input(f"Nombres de variables (separados por espacio, {n} nombres, ej: x y z): ").split()
    out = sistema_a_forma_matricial(A, b, nombres)
    print(out["texto"])

def op7_matriz_por_vector():
    print("\n--- Multiplicación matriz·vector con explicación ---")
    m = int(input("Filas de A (m): "))
    n = int(input("Columnas de A (n): "))
    A = leer_matriz(m, n, "Matriz A:")
    v = leer_vector(n, "Vector columna v de tamaño n: ")
    out = multiplicacion_matriz_vector_explicada(A, v)
    print(out["texto"])

# -------- Menús compactos --------
def menu_sistemas():
    while True:
        print("\n--- Sistemas de ecuaciones ---")
        print("1) Resolver por Gauss-Jordan con pasos")
        print("2) Transformar a forma matricial Ax=b (identificar A, x, b)")
        print("3) Resolver AX=B")
        print("b) Volver")
        op = input("Opción: ").strip().lower()
        if op == "1": op0_resolver_sistema()
        elif op == "2": op6_sistema_a_matriz()
        elif op == "3": op4_ax_igual_b()
        elif op == "b": break
        else: print("Opción inválida.")

def menu_vectores():
    while True:
        print("\n--- Operaciones con vectores ---")
        print("1) Verificar propiedades en R^n")
        print("2) Verificar distributiva A(u+v)=Au+Av")
        print("3) Combinación lineal de vectores")
        print("4) Ecuación vectorial (¿b en span{v1..vk}?)")
        print("b) Volver")
        op = input("Opción: ").strip().lower()
        if op == "1": op1_propiedades()
        elif op == "2": op5_distributiva()
        elif op == "3": op2_combinacion()
        elif op == "4": op3_ecuacion_vectorial()
        elif op == "b": break
        else: print("Opción inválida.")

def op_matriz_vector():
    # opción directa sin submenú
    op7_matriz_por_vector()

def menu():
    while True:
        print("\n=== Calculadora de Álgebra Lineal ===")
        print("1) Sistemas de ecuaciones (resolver, Ax=b, forma matricial)")
        print("2) Operaciones con vectores (propiedades, distributiva, combinación, ecuación vectorial)")
        print("3) Multiplicación matriz·vector (explicada)")
        print("q) Salir")
        op = input("Opción: ").strip().lower()

        if op == "1":
            menu_sistemas()
        elif op == "2":
            menu_vectores()
        elif op == "3":
            op_matriz_vector()
        elif op == "q":
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    menu()
