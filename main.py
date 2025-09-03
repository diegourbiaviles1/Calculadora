from models.Matrix import Matrix
from utils.display import mostrar_sistema_ecuaciones
from models.GaussJordan import gauss_jordan

def main():
    matriz_lista = Matrix.make_matrix(3, 4)
    matriz = Matrix(matriz_lista)   

    print("\n")
    print(matriz)
    
    mostrar_sistema_ecuaciones(matriz)
    
    print("\nSeleccione el método que quiera usar:")
    print("1. Método Escalonado - Eliminación Gaussiana")
    print("2. Salida")
        
    opcion = int(input("R: "))
        
    if opcion == 1:
        gauss_jordan(matriz)
    elif opcion == 2:
        print("Saliendo...")
        return
    else:
        print("Opción no válida.")
    
if __name__ == "__main__":
    main()