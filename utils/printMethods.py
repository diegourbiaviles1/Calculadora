"""
PAra imprimir y registrar pasos de métodos de álgebra lineal.

Funciones diseñadas para ser independientes de los models. Se espera un frame
con la estructura:
dict[step_str] -> {"matrix": list[list[float]], "det": Optional[float], "tag": Optional[str]}
"""

from __future__ import annotations
import copy 
from typing import Any, Callable, Dict, List, Optional

Frame = Dict[str, Dict[str, Any]]

def log_step(frame: Frame, step: str, matrix_state: List[List[float]], det: Optional[float] = None, tag: Optional[str] = None) -> None:
    """Añade un paso al `frame`.
    - frame: diccionario donde se guardan los pasos.
    - step: descripción legible del paso.
    - matrix_state: estado de la matriz (se hace deepcopy internamente).
    - det: valor del determinante (si aplica) o None.
    - tag: etiqueta corta (ej. 'swap', 'elim', 'det').
    """
    frame[step] = {"matrix": copy.deepcopy(matrix_state), "det": det, "tag":tag}
    
    
def _to_str_cell(value: Any, decimals: int=3) -> str:
    try:
        if isinstance(value, float):
            return f"{round(value, decimals)}"
        if isinstance(value, int):
            return str(value)
        return f"{round(float(value), decimals)}"
    except Exception:
        return str(value)
    
def format_matrix(matrix: List[List[float]], decimals: int=3) -> str:
    if not matrix:
        return ""
    
    str_rows: List[List[str]] = [[_to_str_cell(v, decimals) for v in row] for row in matrix]
    cols = len(str_rows[0])
    col_widths = [0] * cols
    for i in str_rows:
        for j in range(cols):
            col_widths[j] = max(col_widths[j], len(i[j]))
            
    lines: List[str] = []
    for r in str_rows:
        padded = [r[i].ljust(col_widths[i]) for i in range(cols)]
        lines.append("  ".join(padded))
    return "\n".join(lines)

def print_frame(frame: Frame, decimals: int=3, show_det: bool = True, sep: str = "-"*40)->None:
    for step, payload in frame.items():
        matrix_state = payload.get("matrix")
        det_val = payload.get("det")
        tag = payload.get("tag")
        print(step)
        if matrix_state:
            print(format_matrix(matrix_state, decimals=decimals))
        if show_det and det_val is not None:
            try:
                print("Determinante:", round(float(det_val), decimals))
            except Exception:
                print("Determinante:", det_val)
        if tag:
            print(f"(tag: {tag})")
        print(sep)
        
def run_and_print(executor: Optional[Callable[[], Frame]] = None, frame: Optional[Frame] = None, decimals:int=3, show_det:bool=True) ->None:
    if frame is None:
        if executor is None:
            raise ValueError("Debe proporcionar executor o frame")
        frame = executor()
    print("\n=== Proceso de Eliminación Gaussiana ===\n")
    print_frame(frame, decimals=decimals, show_det=show_det)
    
__all__ = ["log_step", "format_matrix", "print_frame", "run_and_print"]