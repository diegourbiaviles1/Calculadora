from models.Matrix import Matrix
from utils.display import mostrar_sistema_ecuaciones
from utils.gauss_jordan import gauss_jordan

def main():
    matriz_lista = Matrix.make_matrix(3, 3)
    matriz = Matrix(matriz_lista)    
    print(matriz)
    
    mostrar_sistema_ecuaciones(matriz)
    
if __name__ == "__main__":
    main()