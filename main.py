# main.py
# Nota: Se mantiene el mismo formato de mensajes y menús.
# Cambios clave: validaciones de entrada, manejo de excepciones, y eliminación de import 'info' conflictivo.

from utilidad import (
    leer_matriz, leer_vector, leer_lista_vectores, leer_dimensiones, mat_from_columns
)
from algebra_vector import (
   verificar_propiedades, combinacion_lineal_explicada, ecuacion_vectorial, resolver_AX_igual_B,
    verificar_distributiva_matriz, sistema_a_forma_matricial, multiplicacion_matriz_vector_explicada
)
from sistema_lineal import SistemaLineal, formatear_solucion_parametrica
from homogeneo import (
    analizar_sistema,
    analizar_dependencia,
    resolver_sistema_homogeneo_y_no_homogeneo,
    resolver_dependencia_lineal_con_homogeneo
)
# ---- Programa 5: Operaciones con matrices y traspuesta ----
from matrices import (
    suma_matrices_explicada, resta_matrices_explicada,
    escalar_por_matriz_explicada, multiplicacion_matrices_explicada,
    traspuesta_explicada
)

# ----------------- Helpers de entrada/validación -----------------
def pedir_entero(msg, minimo=None, maximo=None):
    while True:
        try:
            val = int(input(msg))
            if minimo is not None and val < minimo:
                print(f"Valor inválido: debe ser >= {minimo}.")
                continue
            if maximo is not None and val > maximo:
                print(f"Valor inválido: debe ser <= {maximo}.")
                continue
            return val
        except ValueError:
            print("Entrada inválida: escribe un entero.")

def pedir_flotante(msg):
    while True:
        try:
            return float(input(msg))
        except ValueError:
            print("Entrada inválida: escribe un número (float).")

def pedir_opcion(msg, opciones_validas):
    opciones_validas = [o.lower() for o in opciones_validas]
    while True:
        op = input(msg).strip().lower()
        if op in opciones_validas:
            return op
        print("Opción inválida.")

def pedir_si_no(msg):
    return pedir_opcion(msg + " ", {"s", "n"}) == "s"

# ----------------- Operaciones -----------------
def op0_resolver_sistema():
    try:
        print("\n--- Resolver sistema lineal (Gauss-Jordan con pasos) ---")
        m, n = leer_dimensiones("Número de ecuaciones m y número de variables n: ")
        print(f"\nIntroduce la MATRIZ AUMENTADA con {n+1} valores por fila (coeficientes y término independiente).")
        Ab = [leer_vector(n+1, f"Fila {i} (a1 ... an | b): ") for i in range(1, m+1)]
        sl = SistemaLineal(Ab, decimales=4)
        out = sl.gauss_jordan()
        print("\n=== Pasos ===")
        for paso in out["pasos"]:
            print(paso, "\n")
        if out["tipo"] == "unica":
            print("Solución única x =", out["x"])
        elif out["tipo"] == "infinitas":
            print("Infinitas soluciones. Variables libres en columnas:", out["libres"])
        else:
            print("Sistema inconsistente (sin solución).")
        print()
        print(formatear_solucion_parametrica(out, nombres_vars=None, dec=4, fracciones=True))
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def op1_propiedades():
    try:
        print("\n--- Propiedades en R^n ---")
        n = pedir_entero("Dimensión n: ", minimo=1)
        v = leer_vector(n, "v: ")
        u = leer_vector(n, "u: ")
        w = leer_vector(n, "w: ")
        a = pedir_flotante("Escalar a: ")
        b = pedir_flotante("Escalar b: ")
        res = verificar_propiedades(v, u, w, a, b)
        for k, val in res.items():
            print(f"{k}: {val}")
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def op_combinacion_lineal_y_gauss_jordan():
    try:
        print("\n--- Combinación lineal ---")
        k = pedir_entero("Cantidad de vectores k: ", minimo=1)
        n = pedir_entero("Dimensión n: ", minimo=1)
        V = leer_lista_vectores(k, n)
        coef = leer_vector(k, "Coeficientes (c1..ck): ")

        print("\nCalculando combinación lineal...")
        out_combinacion = combinacion_lineal_explicada(V, coef)
        print("\n--- Resultado de la combinación lineal ---")
        print(out_combinacion["texto"])
        print("\nComo lista:", out_combinacion["resultado_simple"])

        print("\n--- Resolviendo por Gauss-Jordan ---")
        A = mat_from_columns(V)
        b = out_combinacion["resultado"]
        matriz_aumentada = [A[i] + [b[i]] for i in range(n)]

        sl = SistemaLineal(matriz_aumentada, decimales=4)
        resultado_gauss_jordan = sl.gauss_jordan()

        print("\n--- Proceso de Gauss-Jordan ---")
        for paso in resultado_gauss_jordan["pasos"]:
            print(paso)

        print("\nResultado de Gauss-Jordan:")
        if resultado_gauss_jordan["tipo"] == "unica":
            print("c (coeficientes) =", resultado_gauss_jordan["x"])
        elif resultado_gauss_jordan["tipo"] == "infinitas":
            print("El sistema tiene infinitas soluciones.")
            print("Columnas pivote:", resultado_gauss_jordan["pivotes"])
            print("Variables libres (columnas):", resultado_gauss_jordan["libres"])
        else:
            print("Sistema inconsistente (sin solución).")

        print()
        print(formatear_solucion_parametrica(resultado_gauss_jordan, nombres_vars=None, dec=4, fracciones=True))
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def op3_ecuacion_vectorial():
    try:
        print("\n--- Ecuación vectorial (¿b en span{v1..vk}?) ---")
        k = pedir_entero("Cantidad de vectores k: ", minimo=1)
        n = pedir_entero("Dimensión n: ", minimo=1)
        V = leer_lista_vectores(k, n)
        b = leer_vector(n, "b: ")
        out = ecuacion_vectorial(V, b)
        for paso in out.get("reportes", []):
            print(paso, "\n")
        tipo = out.get("tipo") or out.get("estado")
        print("Estado:", tipo)
        print()
        print(formatear_solucion_parametrica(out, nombres_vars=None, dec=4, fracciones=True))
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def op4_ax_igual_b():
    try:
        print("\n--- Ecuación matricial AX=B ---")
        m, n = leer_dimensiones("Tamaño de A (m n): ")
        A = leer_matriz(m, n, "Matriz A:")
        es_matriz_B = pedir_si_no("¿B tiene varias columnas? (s/n):")
        if es_matriz_B:
            p = pedir_entero("Número de columnas de B (p): ", minimo=1)
            B = leer_matriz(m, p, "Matriz B:")
        else:
            B = leer_vector(m, "Vector b: ")
        out = resolver_AX_igual_B(A, B)
        for paso in out.get("reportes", []):
            print(paso, "\n")
        if out.get("estado") == "ok":
            if "x" in out:
                print("x =", out["x"])
            if "X" in out:
                print("X =\n", out["X"])
        else:
            print("Estado:", out.get("estado"))
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def op5_distributiva():
    try:
        print("\n--- Verificar distributiva A(u+v) = Au + Av ---")
        m = pedir_entero("Filas de A (m): ", minimo=1)
        n = pedir_entero("Columnas de A (n): ", minimo=1)
        A = leer_matriz(m, n, "Matriz A:")
        u = leer_vector(n, "u (vector columna de tamaño n): ")
        v = leer_vector(n, "v (vector columna de tamaño n): ")
        out = verificar_distributiva_matriz(A, u, v)
        for p in out["pasos"]:
            print(p)
        print("\n" + out["conclusion"])
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def op6_sistema_a_matriz():
    try:
        print("\n--- Sistema → forma matricial Ax = b ---")
        m = pedir_entero("Número de ecuaciones (m): ", minimo=1)
        n = pedir_entero("Número de variables (n): ", minimo=1)
        print("Introduce la matriz de coeficientes A (m x n):")
        A = leer_matriz(m, n)
        print("Introduce el vector de términos independientes b (m):")
        b = leer_vector(m, "b: ")
        while True:
            nombres = input(f"Nombres de variables (separados por espacio, {n} nombres, ej: x y z): ").split()
            if len(nombres) == n:
                break
            print(f"Debes ingresar exactamente {n} nombres.")
        out = sistema_a_forma_matricial(A, b, nombres)
        print(out["texto"])
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def op7_matriz_por_vector():
    try:
        print("\n--- Multiplicación matriz·vector con explicación ---")
        m = pedir_entero("Filas de A (m): ", minimo=1)
        n = pedir_entero("Columnas de A (n): ", minimo=1)
        A = leer_matriz(m, n, "Matriz A:")
        v = leer_vector(n, "Vector columna v de tamaño n: ")
        out = multiplicacion_matriz_vector_explicada(A, v)
        print(out["texto"])
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

# -------- Programa 4 --------
def opP4_sistema_h_oh():
    try:
        print("\n--- Programa 4: Sistema homogéneo / no homogéneo + Dependencia lineal ---")
        m, n = leer_dimensiones("Tamaño de A (m n): ")
        A = leer_matriz(m, n, "Matriz A:")
        b = leer_vector(m, "Vector b: ")

        # Parte 1: Ax = b
        info_res = resolver_sistema_homogeneo_y_no_homogeneo(A, b)

        print("\n=== PASOS (Gauss-Jordan aplicado a Ax = b) ===")
        for paso in info_res["pasos"]:
            print(paso)

        print("\n=== SOLUCIÓN GENERAL (forma paramétrica) ===")
        print(info_res["salida_parametrica"])

        print("\n=== CONCLUSIÓN DEL SISTEMA ===")
        print(info_res["conclusion"])

        # Parte 2: A·c = 0 (dependencia)
        info_dep = resolver_dependencia_lineal_con_homogeneo(A)

        print("\n\n=== ANÁLISIS DE DEPENDENCIA LINEAL ===")
        print(info_dep["dependencia"])

        print("\n=== PASOS (Gauss-Jordan aplicado a A·c = 0) ===")
        for paso in info_dep["pasos"]:
            print(paso)

        print("\n=== COMBINACIÓN LINEAL (forma paramétrica de los coeficientes c) ===")
        print(info_dep["salida_parametrica"])
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def opP4_dependencia():
    try:
        print("\n--- Programa 4: Dependencia / Independencia lineal ---")
        k = pedir_entero("Cantidad de vectores k: ", minimo=1)
        n = pedir_entero("Dimensión n: ", minimo=1)
        V = leer_lista_vectores(k, n)
        info = analizar_dependencia(V)

        print("\n=== Conclusion ===")
        print(info["mensaje"])

        print("\n=== Pasos (Gauss-Jordan sobre A c = 0) ===")
        for p in info["pasos"]:
            print(p, "\n")

        print("\n=== Combinación paramétrica de coeficientes c ===")
        print(info["salida_parametrica"])
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

# -------- Programa 5: Operaciones con matrices y traspuesta --------
# -------- Programa 5: Operaciones con matrices y traspuesta --------
def opP5_suma():
    try:
        print("\n--- Programa 5: Suma de matrices (con verificación de traspuestas) ---")
        m, n = leer_dimensiones("Dimensiones de A y B (m n): ")
        A = leer_matriz(m, n, "Matriz A:")
        B = leer_matriz(m, n, "Matriz B:")
        out = suma_matrices_explicada(A, B)

        print("\nInterpretación:", out["mensaje"])
        print("\n--- Pasos ---")
        for p in out["pasos"]:
            print(p)
        print("\n--- Verificación de propiedad ---")
        print(out["propiedad"])
        print(out["detalle_verificacion"])
        input("\nPresiona ENTER para continuar...")
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")


def opP5_resta():
    try:
        print("\n--- Programa 5: Resta de matrices (con verificación de traspuestas) ---")
        m, n = leer_dimensiones("Dimensiones de A y B (m n): ")
        A = leer_matriz(m, n, "Matriz A:")
        B = leer_matriz(m, n, "Matriz B:")
        out = resta_matrices_explicada(A, B)

        print("\nInterpretación:", out["mensaje"])
        print("\n--- Pasos ---")
        for p in out["pasos"]:
            print(p)
        print("\n--- Verificación de propiedad ---")
        print(out["propiedad"])
        print(out["detalle_verificacion"])
        input("\nPresiona ENTER para continuar...")
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")


def opP5_escalar():
    try:
        print("\n--- Programa 5: Multiplicación por escalar (con verificación de traspuestas) ---")
        m, n = leer_dimensiones("Dimensiones de A (m n): ")
        A = leer_matriz(m, n, "Matriz A:")
        k = pedir_flotante("Escalar k: ")
        out = escalar_por_matriz_explicada(k, A)

        print("\nInterpretación:", out["mensaje"])
        print("\n--- Pasos ---")
        for p in out["pasos"]:
            print(p)
        print("\n--- Verificación de propiedad ---")
        print(out["propiedad"])
        print(out["detalle_verificacion"])
        input("\nPresiona ENTER para continuar...")
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")



def opP5_producto():
    try:
        print("\n--- Programa 5: Multiplicación de matrices AB ---")
        m, n = leer_dimensiones("Dimensiones de A (m n): ")
        A = leer_matriz(m, n, "Matriz A:")
        n2, p = leer_dimensiones("Dimensiones de B (n p): ")
        if n2 != n:
            print("Advertencia: Debe cumplirse columnas(A)=filas(B). Si no, fallará.")
        B = leer_matriz(n2, p, "Matriz B:")
        out = multiplicacion_matrices_explicada(A, B)
        print("\nInterpretación:", out["mensaje"])
        print("\n--- Pasos ---")
        for ptxt in out["pasos"]:
            print(ptxt)
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def opP5_traspuesta():
    try:
        print("\n--- Programa 5: Traspuesta de A (con verificación de propiedades) ---")
        m, n = leer_dimensiones("Dimensiones de A (m n): ")
        A = leer_matriz(m, n, "Matriz A:")

        usar_B = pedir_si_no("¿Deseas ingresar una matriz B para verificar propiedades como (A+B)^T y (AB)^T? (s/n):")
        B = None
        if usar_B:
            mB, nB = leer_dimensiones("Dimensiones de B (m n): ")
            B = leer_matriz(mB, nB, "Matriz B:")

        usar_k = pedir_si_no("¿Deseas ingresar un escalar k para verificar (kA)^T = kA^T? (s/n):")
        k = pedir_flotante("Escalar k: ") if usar_k else None

        out = traspuesta_explicada(A, B=B, k=k)
        print("\nInterpretación:", out["mensaje"])
        print("\n--- Pasos ---")
        for p in out["pasos"]:
            print(p)
        print("\n--- Propiedades verificadas ---")
        for pr in out["propiedades"]:
            print("•", pr)
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"Error: {e}")

def menu_programa5():
    while True:
        print("\n--- Programa 5: Operaciones con matrices y traspuesta ---")
        print("1) Suma de matrices")
        print("2) Resta de matrices")
        print("3) Multiplicación por escalar")
        print("4) Multiplicación de matrices (AB)")
        print("5) Traspuesta (con propiedades)")
        print("b) Volver")
        op = pedir_opcion("Opción: ", {"1","2","3","4","5","b"})
        if op == "1":
            opP5_suma()
        elif op == "2":
            opP5_resta()
        elif op == "3":
            opP5_escalar()
        elif op == "4":
            opP5_producto()
        elif op == "5":
            opP5_traspuesta()
        elif op == "b":
            break

# -------- Menús --------
def menu_sistemas():
    while True:
        print("\n--- Sistemas de ecuaciones ---")
        print("1) Resolver por Gauss-Jordan con pasos")
        print("2) Transformar a forma matricial Ax=b (identificar A, x, b)")
        print("3) Resolver AX=B")
        print("b) Volver")
        op = pedir_opcion("Opción: ", {"1", "2", "3", "b"})
        if op == "1":
            op0_resolver_sistema()
        elif op == "2":
            op6_sistema_a_matriz()
        elif op == "3":
            op4_ax_igual_b()
        elif op == "b":
            break

def menu_vectores():
    while True:
        print("\n--- Operaciones con vectores ---")
        print("1) Verificar propiedades en R^n")
        print("2) Verificar distributiva A(u+v)=Au+Av")
        print("3) Combinación lineal de vectores")
        print("4) Ecuación vectorial (¿b en span{v1..vk}?)")
        print("b) Volver")
        op = pedir_opcion("Opción: ", {"1", "2", "3", "4", "b"})
        if op == "1":
            op1_propiedades()
        elif op == "2":
            op5_distributiva()
        elif op == "3":
            op_combinacion_lineal_y_gauss_jordan()
        elif op == "4":
            op3_ecuacion_vectorial()
        elif op == "b":
            break

def op_matriz_vector():
    op7_matriz_por_vector()

def menu():
    try:
        while True:
            print("\n=== Calculadora ===")
            print("1) Sistemas de ecuaciones (resolver, Ax=b, forma matricial)")
            print("2) Operaciones con vectores (propiedades, distributiva, combinación, ecuación vectorial)")
            print("3) Multiplicación matriz·vector (explicada)")
            print("4) Sistema homogéneo / no homogéneo + Dependencia lineal")
            print("5) Operaciones con matrices y traspuesta (Programa 5)")
            print("q) Salir")
            op = pedir_opcion("Opción: ", {"1", "2", "3", "4", "5", "q"})
            if op == "1":
                menu_sistemas()
            elif op == "2":
                menu_vectores()
            elif op == "3":
                op_matriz_vector()
            elif op == "4":
                opP4_sistema_h_oh()
            elif op == "5":
                menu_programa5()
            elif op == "q":
                break
    except KeyboardInterrupt:
        print("\nSalida solicitada por el usuario.")

if __name__ == "__main__":
    menu()
