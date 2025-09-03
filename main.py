from models.Matrix import Matrix
from methods.gauss_jordan import gauss_jordan

def main():
    matriz_lista = Matrix.make_matrix(3, 3)
    matriz = Matrix(matriz_lista)
    print(matriz)
    
if __name__ == "__main__":
    main()