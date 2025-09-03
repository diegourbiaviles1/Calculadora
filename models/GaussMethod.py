from models.Matrix import Matrix
from math import isclose
import copy

class GaussMethod(Matrix):
    def __init__(self, matriz: list[list]) -> None:
        super().__init__(copy.deepcopy(matriz))
        self.determinante = 1
        self.tolerance = 1e-13 # Notacion cientifica
        self.frame = {} # Registro de pasos en forma de diccionario
        
    
    def gauss_method(self):
        if self.filas != self.columnas:
            return False
        if not self.null_determinant():
            for col in range(self.columnas):
                self.triangular_steps(col)
                self.triangular_reduction(col)
            self.determinant_result()
        return self.frame
    
    def null_determinant(self) -> bool:
        # Validar Matrices singulares por lineas o columnas 0
        for row in range(self.filas):
            if all(self.matriz[row][i]==0 for i in range(self.columnas)):
                self.frame[f"Fila {row+1} son 0, su determinante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                return True
            
        for col in range(self.columnas):
            if all(self.matriz[i][col]==0 for i in range(self.filas)):
                self.frame[f"Columna {col+1} son 0, su determinante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                return True
            
        # Validar Matrices singulares por repeticion de filas o columnas
        for i in range(self.filas):
            for j in range(i+1, self.filas):
                if self.matriz[i] == self.matriz[j]:
                    self.frame[f"Filas {i+1} y {j+1} son iguales, su determinmante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                    return True
                
        for i in range(self.columnas):
            for j in range(i+1, self.columnas):
                col_i = [self.matriz[k][i] for k in range(self.filas)]
                col_j = [self.matriz[k][j] for k in range(self.filas)]
                if col_i == col_j:
                    self.frame[f"Columnas {i+1} y {j+1} son iguales, su determinante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                    return True
        
        return False
    
    def determinant_result(self):
        opr = f"({self.determinante})"
        for i in range(self.filas):
            opr += f"({self.matriz[i][i]:.1f})"
            self.determinante *= self.matriz[i][i]
        results = f"Determinante = {opr}"
        self.frame[results] = (copy.deepcopy(self.matriz), f"{round(self.determinante, 4)}")
        
        
    def swipe_rows(self, fila_a, fila_b) -> None:
        """
        Intercambia las filas de la matriz, actualiza el determinante y
        guarda el estado en self.frame
        """
        temp = self.matriz[fila_a]
        self.matriz[fila_a] = self.matriz[fila_b]
        self.matriz[fila_b] = temp
        # Cambiar el signo del determinante
        self.determinante *= -1
        # Guardar la matri
        det = f"Fila {fila_a + 1} <--> Fila {fila_b + 1}"
        self.frame[det] = (copy.deepcopy(self.matriz), f"{self.determinante}")
        
    def swipe_col(self, col_a, col_b) -> None:
        """
        Intercambia las columnas de la matriz, actualiza el determinante
        y guarda el estado en self.frame
        """
        for fila in range(self.filas):
            temp = self.matriz[fila][col_a]
            self.matriz[fila][col_a] = self.matriz[fila][col_b]
            self.matriz[fila][col_b] = temp
            
        # Cambiar el determinante
        self.determinante *= -1
        #Guardar matrriz
        det = f"Columna {col_a + 1} <--> Columna {col_b + 1}"
        self.frame[det] = (copy.deepcopy(self.matriz), f"{self.determinante}")
    
    
    def triangular_reduction(self, col):
        pivote = self.matriz[col][col]
        for fila in range(col + 1, self.filas):
            valor_actual = self.matriz[fila][col]
            if isclose(valor_actual, 0, abs_tol=self.tolerance):
                continue
            
            factor = valor_actual / pivote
            nueva_fila = []
            for i in range(self.columns):
                nuevo_valor = self.matriz[fila][i] - factor * self.matriz[col][i]
                '''Si el valor es cercano a 0, se deja exactamente en 0'''
                if isclose(nuevo_valor, 0, abs_tol=self.tolerance):
                    nuevo_valor = 0
                nueva_fila.append(nuevo_valor)
            self.matriz[fila] = nueva_fila
            
            # Guardar el registro
            operator = "-" if factor > 0 else "+"
            factor_abs = abs(factor)
            self.frame[f"F{fila+1} -> F{fila+1} {operator} {factor_abs:.3f}*F{col+1}"] = (copy.deepcopy(self.matriz), f"{self.determinante}")
    
    def triangular_steps(self, col:int):
        """
        Busca el mejor pivote en la columna 'col' (valor más cercano a 1 y distinto de 0).
        Si no lo encuentra en la columna, busca en otras columnas de la misma fila.
        Realiza el intercambio de filas o columnas si es necesario.
        Guarda el paso en self.frame si no hay pivote.
        """
        row_option = -1
        col_option = col
        distance_option = float('inf')
        
        for fila in range(col, self.filas):
            valor = self.matriz[fila][col]
            if isclose(valor, 0, abs_tol=self.tolerance):
                continue
            distance = abs(valor -1)
            if isclose(valor, 1, abs_tol=self.tolerance):
                row_option = fila
                distance_option = 0
                break
            elif distance < distance_option:
                distance_option = distance
                row_option = fila
                
        '''Si no hay pivote 1 en la columna, buscar en otras columnas'''
        if row_option != -1 and not isclose(self.matriz[row_option][col], 1, abs_tol=self.tolerance):
            for col_alt in range(col+1, self.columnas):
                valor = self.matriz[col][col_alt]
                if isclose(valor, 0, abs_tol=self.tolerance):
                    continue
                distance = abs(valor-1)
                if isclose(valor, 1, abs_tol=self.tolerance):
                    col_option = col_alt
                    distance_option = 0
                    break
                elif distance < distance_option:
                    distance_option = distance
                    col_option = col_alt
                    
        # Si no hay pivote
        if row_option == -1:
            self.frame[f"No hay pivote en la columna {col+1}."]=(copy.deepcopy(self.matriz), f"{self.determinante}")
            return
        
        # Intercambiar columna
        if col_option != col:
            self.swipe_col(col, col_option)
            return
        
        if row_option != col:
            self.swipe_rows(col, row_option)
            return
    
    def __str__(self) -> str:
        return super().__str__()