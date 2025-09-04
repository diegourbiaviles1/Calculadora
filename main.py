from models.Matrix import Matrix
from utils.display import mostrar_sistema_ecuaciones
from models.GaussJordan import gauss_jordan
from models.GaussMethod import GaussMethod
from utils.printMethods import print_frame

def main():
    matriz_lista = Matrix.make_matrix(3, 3)
    matriz = Matrix(matriz_lista)
    gauss = GaussMethod(matriz_lista)
    gauss.gauss_method()
    print_frame(gauss.frame)
    
if __name__ == "__main__":
    main()