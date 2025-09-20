# main.py
from sistema_lineal import SistemaLineal
from utilidad import leer_dimensiones, leer_matriz, leer_vector, leer_lista_vectores
from algebra_vector import verificar_propiedades, combinacion_lineal, ecuacion_vectorial, resolver_AX_igual_B

def op0_resolver_sistema():
    print("\n--- Resolver sistema lineal (Gauss-Jordan con pasos) ---")
    m, n = leer_dimensiones("Número de ecuaciones m y número de variables n: ")
    print(f"\nIntroduce la MATRIZ AUMENTADA con {n+1} valores por fila (coeficientes y término independiente).")
    Ab = []
    for i in range(1, m+1):
        fila = leer_vector(n+1, f"Fila {i} (a1 ... an | b): ")
        Ab.append(fila)
    sistema = SistemaLineal(Ab, decimales=4)
    print("\n=== Resolviendo ===\n")
    print(sistema.eliminacion_gaussiana())

def op1_propiedades():
    print("\n--- 1) Propiedades algebraicas en R^n ---")
    n = int(input("¿Tamaño de los vectores n?: ").strip())
    v = leer_vector(n, "v: ")
    u = leer_vector(n, "u: ")
    w = leer_vector(n, "w: ")
    a = float(input("Escalar a: ").strip())
    b = float(input("Escalar b: ").strip())

    # Mostrar operaciones solicitadas
    from utilidad import vec_suma, escalar_por_vector
    print("\nOperaciones explícitas:")
    print("v + u =", vec_suma(v, u))
    print("a · u =", escalar_por_vector(a, u))
    print("b · v =", escalar_por_vector(b, v))
    print("-v    =", escalar_por_vector(-1.0, v))
    print("0⃗    =", [0.0]*n)

    # Verificación de propiedades
    from algebra_vector import verificar_propiedades
    res = verificar_propiedades(v, u, w, a, b)
    print("\nVerificación de propiedades:")
    for k, ok in res.items():
        print(f"  {k}: {'Se cumple' if ok else 'NO se cumple'}")


def op2_combinacion_lineal():
    print("\n--- 2) Combinación lineal de vectores ---")
    n = int(input("Dimensión n: ").strip())
    k = int(input("Cantidad de vectores k: ").strip())
    vectores = leer_lista_vectores(k, n)
    b = leer_vector(n, "Vector objetivo b: ")
    info = combinacion_lineal(vectores, b, decimales=4)
    print("\nDiagnóstico:", 
          "NO pertenece al span" if info["estado"]=="no_pertenece"
          else "Solución única" if info["estado"]=="unica"
          else "Infinitas soluciones" if info["estado"]=="infinitas"
          else info["estado"])
    if info["coeficientes"] is not None:
        print("Coeficientes c que cumplen A c = b:", info["coeficientes"])
    print("\n--- Procedimiento ---\n", info["traza"])

def op3_ecuacion_vectorial():
    print("\n--- 3) Ecuación vectorial c1 v1 + ... + ck vk = b ---")
    n = int(input("Dimensión n: ").strip())
    k = int(input("Cantidad de vectores k: ").strip())
    vectores = leer_lista_vectores(k, n)
    b = leer_vector(n, "b: ")
    info = ecuacion_vectorial(vectores, b, decimales=4)
    print("\nDiagnóstico:", 
          "NO tiene solución" if info["estado"]=="no_pertenece"
          else "Solución única" if info["estado"]=="unica"
          else "Infinitas soluciones" if info["estado"]=="infinitas"
          else info["estado"])
    if info["coeficientes"] is not None:
        print("Coeficientes (c1..ck):", info["coeficientes"])
    print("\n--- Procedimiento ---\n", info["traza"])

def op4_ecuacion_matricial():
    print("\n--- 4) Ecuación matricial AX = B ---")
    m, n = leer_dimensiones("Dimensiones de A (m n): ")
    A = leer_matriz(m, n, "Introduce A fila por fila:")
    print("\n¿B es vector (1) o matriz (2)?")
    tipo = input("Elige 1/2: ").strip()
    if tipo == "1":
        B = leer_vector(m, "B (vector de tamaño m): ")
    else:
        mp, p = leer_dimensiones("Dimensiones de B (m p): ")
        if mp != m:
            print("B debe tener el mismo número de filas que A (m).")
            return
        B = leer_matriz(m, p, "Introduce B fila por fila:")
    out = resolver_AX_igual_B(A, B, decimales=4)
    print("\nEstado:", out["estado"])
    print("X:", out["X"])
    print("\n--- Procedimiento por columna de B ---")
    for idx, rep in enumerate(out["reportes"], start=1):
        print(f"\n[B columna {idx}]\n{rep}")

def menu():
    while True:
        print("\n=== Calculadora de Álgebra Lineal ===")
        print("0) Resolver sistema lineal (Gauss-Jordan con pasos) ")
        print("1) Propiedades en R^n")
        print("2) Combinación lineal de vectores")
        print("3) Ecuación vectorial")
        print("4) Ecuación matricial AX=B")
        print("q) Salir")
        op = input("Opción: ").strip().lower()
        if   op == "0": op0_resolver_sistema()
        elif op == "1": op1_propiedades()
        elif op == "2": op2_combinacion_lineal()
        elif op == "3": op3_ecuacion_vectorial()
        elif op == "4": op4_ecuacion_matricial()
        elif op == "q": break
        else: print("Opción no válida.")

if __name__ == "__main__":
    menu()
