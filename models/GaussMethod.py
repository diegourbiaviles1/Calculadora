from models.Matrix import Matrix
from math import isclose
import copy
from typing import Any, Dict, List, Optional
from utils.printMethods import log_step
from utils.EquationFunctions import format_number

"""
Metodo de Gauss con:
    - Soporte para matrices rectangulares (m x n).
    - Calculo de determinante SOLO si la matriz es cuadrada.
    - Registro de pasos estandarizado en self.frame (dict de pasos -> {matrix, det, tag}).
    - Métodos de impresión del proceso: print_process() y run_and_print().
"""
class GaussMethod(Matrix):
    def __init__(self, matriz: list[list]) -> None:
        super().__init__(copy.deepcopy(matriz))
        self.determinante = 1
        self.tolerance = 1e-13 # Notacion cientifica
        self.frame: Dict[str, Dict[str, Any]] = {}
        self.analize: Dict[str, Any] = {}
    
    # ----------------------
    # MÉTODO PRINCIPAL
    # ----------------------
    def gauss_method(self) -> dict[str, dict[str, any]]:
        self.frame = {}
        self.determinante = 1.0
        
        max_steps = min(self.filas, self.columnas)
        if self.filas == self.columnas and self.null_determinant():
            return self.frame
        
        if not self.null_determinant():
            for col in range(max_steps):
                self.pivote(col)
                self.triangular_reduction(col)
            if self.filas == self.columnas:
                self.determinant_result()
        return self.frame

    # ----------------------
    # CHEQUEOS DE SINGULARIDAD (solo para cuadradas)
    # ----------------------
    def null_determinant(self) -> bool:
        if self.filas != self.columnas:
            return False
        
        # Validar Matrices singulares por lineas o columnas 0
        for row in range(self.filas):
            if all(self.matriz[row][i]==0 for i in range(self.columnas)):
                log_step(self.frame, f"Fila {row+1} son 0, su determinante es 0 (Matriz Singular)", self.matriz, det=0, tag="singular")
                #self.frame[f"Fila {row+1} son 0, su determinante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                return True
            
        for col in range(self.columnas):
            if all(self.matriz[i][col]==0 for i in range(self.filas)):
                log_step(self.frame, f"Columna {col+1} son 0, su determinante es 0 (Matriz Singular)", self.matriz, det=0, tag="singular")
                #self.frame[f"Columna {col+1} son 0, su determinante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                return True
            
        # Validar Matrices singulares por repeticion de filas o columnas
        for i in range(self.filas):
            for j in range(i+1, self.filas):
                if self.matriz[i] == self.matriz[j]:
                    log_step(self.frame, f"Filas {i+1} y {j+1} son iguales, su determinmante es 0 (Matriz Singular)", self.matriz, det=0, tag="singular")
                    #self.frame[f"Filas {i+1} y {j+1} son iguales, su determinmante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                    return True
                
        for i in range(self.columnas):
            for j in range(i+1, self.columnas):
                col_i = [self.matriz[k][i] for k in range(self.filas)]
                col_j = [self.matriz[k][j] for k in range(self.filas)]
                if col_i == col_j:
                    log_step(self.frame, f"Columnas {i+1} y {j+1} son iguales, su determinante es 0 (Matriz Singular)", self.matriz, det=0, tag="singular")
                    #self.frame[f"Columnas {i+1} y {j+1} son iguales, su determinante es 0 (Matriz Singular)"] = (copy.deepcopy(self.matriz),0)
                    return True
        
        return False
    
    def determinant_result(self):
        if self.filas != self.columnas:
            return 
        
        diag_product = 1.0
        limit = min(self.filas, self.columnas)
        for i in range(limit):
            diag_product *= self.matriz[i][i]
            
        final_dect = self.determinante * diag_product
        details = f"Determinante = ({format_number(self.determinante)})" + "".join(f"({format_number(self.matriz[i][i])})" for i in range(limit))
        log_step(self.frame, details, self.matriz, det=format_number(float(final_dect)), tag="det")
        
    # ---------------------
    # OPERACIONES ELEMENTALES
    # ----------------------
    def swipe_rows(self, fila_a, fila_b) -> None:
        """
        Intercambia las filas de la matriz, actualiza el determinante y
        guarda el estado en self.frame
        """
        temp = self.matriz[fila_a]
        self.matriz[fila_a] = self.matriz[fila_b]
        self.matriz[fila_b] = temp
        # Cambiar el signo del determinante
        if self.filas == self.columnas:
            self.determinante *= -1.0
        # Guardar la matriz
        det = f"Fila {fila_a + 1} <--> Fila {fila_b + 1}"
        log_step(self.frame, f"Fila {fila_a+1} <-> Fila {fila_b+1}", self.matriz,
                det=(format_number(self.determinante) if self.filas == self.columnas else None), tag="swap_rows")

        
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
        if self.filas == self.columnas:
            self.determinante *= -1.0
        #Guardar matrriz
        det = f"Columna {col_a + 1} <--> Columna {col_b + 1}"
        log_step(self.frame, f"Columna {col_a+1} <-> Columna {col_b+1}", self.matriz,
                det=(format_number(self.determinante) if self.filas == self.columnas else None), tag="swap_cols")
    
    
    def triangular_reduction(self, col):
        if col >= self.filas or col >= self.columnas:
            return
        
        pivote = self.matriz[col][col]
        if isclose(pivote, 0.0, abs_tol=self.tolerance):
            log_step(self.frame, f"Pivote ≈ 0 en columna {col+1}, no se reduce", self.matriz,
                    det=(format_number(self.determinante) if self.filas == self.columnas else None), tag="no_pivot_for_reduction")
            return
        
        log_step(self.frame,
            f"Pivote usado para reducción: A[{col+1},{col+1}] = {format_number(pivote)}", self.matriz,
            det=(format_number(self.determinante) if self.filas == self.columnas else None), tag="pivot_use")
        
        for fila in range(col + 1, self.filas):
            valor_actual = self.matriz[fila][col]
            if isclose(valor_actual, 0.0, abs_tol=self.tolerance):
                continue
            
            factor = valor_actual / pivote
            nueva_fila = []
            for i in range(self.columnas):
                nuevo_valor = self.matriz[fila][i] - factor * self.matriz[col][i]
                '''Si el valor es cercano a 0, se deja exactamente en 0'''
                if isclose(nuevo_valor, 0.0, abs_tol=self.tolerance):
                    nuevo_valor = 0.0
                nueva_fila.append(nuevo_valor)
            self.matriz[fila] = nueva_fila
            
            # Guardar el registro
            log_step(self.frame,f"F{fila+1} -> F{fila+1} - {format_number(factor)}*F{col+1}",
                    self.matriz, det=(format_number(self.determinante) if self.filas == self.columnas else None),
                    tag="elim")
    
    def pivote(self, col:int):
        """
        Busca el mejor pivote en la columna 'col' (valor más cercano a 1 y distinto de 0).
        Si no lo encuentra en la columna, busca en otras columnas de la misma fila.
        Realiza el intercambio de filas o columnas si es necesario.
        Guarda el paso en self.frame si no hay pivote.
        """
        if col >= self.filas or col >= self.columnas:
            return
        
        row_option = -1
        col_option = col
        distance_option = float('inf')
        
        for fila in range(col, self.filas):
            valor = self.matriz[fila][col]
            if isclose(valor, 0.0, abs_tol=self.tolerance):
                continue
            distance = abs(valor -1.0)
            if isclose(valor, 1.0, abs_tol=self.tolerance):
                row_option = fila
                distance_option = 0.0
                break
            elif distance < distance_option:
                distance_option = distance
                row_option = fila
                
        '''Si no hay pivote 1 en la columna, buscar en otras columnas'''
        if row_option != -1 and not isclose(self.matriz[row_option][col], 1.0, abs_tol=self.tolerance):
            for col_alt in range(col+1, self.columnas):
                valor = self.matriz[col][col_alt]
                if isclose(valor, 0.0, abs_tol=self.tolerance):
                    continue
                distance = abs(valor-1.0)
                if isclose(valor, 1.0, abs_tol=self.tolerance):
                    col_option = col_alt
                    distance_option = 0.0
                    break
                elif distance < distance_option:
                    distance_option = distance
                    col_option = col_alt
                    
        # Si no hay pivote
        if row_option == -1:
            log_step(self.frame, f"No hay pivote en la columna {col+1}.", self.matriz,
                    det=(format_number(self.determinante) if self.filas == self.columnas else None), tag="no_pivot")
            return
        
        pivot_val = self.matriz[row_option][col_option]
        log_step(self.frame, f"Pivote elegido: A[{row_option+1},{col_option+1}] = {format_number(pivot_val)}",
            self.matriz, det=(format_number(self.determinante) if self.filas == self.columnas else None), tag="pivot")
        
        # Intercambiar columna
        if col_option != col:
            self.swipe_col(col, col_option)
            return
        
        if row_option != col:
            self.swipe_rows(col, row_option)
            return
    
    def __str__(self) -> str:
        return super().__str__()