from models.Matrix import Matrix
from utils.display import mostrar_sistema_ecuaciones
from models.GaussJordan import gauss_jordan
from models.GaussMethod import GaussMethod
from utils.printMethods import print_frame, print_equation

def main():
    #matriz, vector = Matrix.make_equation(3, 3)
    matriz_lista = Matrix.make_matrix(3, 3)
    matriz = Matrix(matriz_lista)
    gauss = GaussMethod(matriz_lista)
    gauss.gauss_method()
    print_frame(gauss.frame)
    #print(print_equation(matriz, vector))
    
if __name__ == "__main__":
    main()