# GUI_TabMetodosNumericos.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# --- Importaciones para Matplotlib (Gráficos) ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from widgets import LabeledEdit, OutputArea, btn
from metodos_numericos import (
    biseccion_descriptiva,
    regla_falsa_descriptiva,
    newton_raphson_descriptiva,
    secante_descriptiva,
    generar_reporte_paso_a_paso,  # Para métodos cerrados
    generar_reporte_abierto,      # Para Newton y Secante
    calcular_errores,
    tipos_de_error_texto,
    ejemplo_punto_flotante_texto,
    propagacion_error
)
from notacion_posicional import descomponer_base10, descomponer_base2
from utilidad import fmt_number, DEFAULT_DEC

# Símbolo global para Sympy
x = sp.symbols("x")

# ==================================================
# 1. Clase para la Ventana del Gráfico
# ==================================================
class VentanaGrafica(QtWidgets.QDialog):
    """
    Ventana emergente con estilo profesional para gráficas matemáticas.
    """
    def __init__(self, expresion_str, x_range, puntos_interes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Análisis Gráfico: {expresion_str}")
        self.resize(900, 650)
        
        # Aplicar estilo visual limpio
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            pass  # Si no está disponible, usa el default

        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._graficar_profesional(expresion_str, x_range, puntos_interes)

    def _graficar_profesional(self, expr_str, x_range, puntos_interes):
        ax = self.figure.add_subplot(111)
        
        # 1. Preparar datos
        expr_clean = expr_str.replace("^", "**").replace("e^", "exp")
        
        try:
            expr_sym = sp.sympify(expr_clean)
            f_np = sp.lambdify(x, expr_sym, modules=['numpy'])
            
            x_min, x_max = x_range
            dist = abs(x_max - x_min)
            if dist < 1e-9: dist = 1.0
            
            # Margen dinámico para mejor visualización
            margin = dist * 0.25
            t = np.linspace(x_min - margin, x_max + margin, 600)
            y = f_np(t)

            # 2. Configuración de Ejes (Estilo Matemático)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            
            # Flechas en los ejes (opcional, requiere ajustes manuales a veces)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            # 3. Graficar la función
            # Línea principal sólida y elegante
            line, = ax.plot(t, y, label=f"$f(x)={expr_str}$", color="#1f77b4", linewidth=2.5, zorder=2)
            
            # Sombra suave bajo la curva (opcional, da profundidad)
            ax.fill_between(t, y, 0, where=(y > 0), color='#1f77b4', alpha=0.1)
            ax.fill_between(t, y, 0, where=(y < 0), color='#ff7f0e', alpha=0.05)

            # 4. Puntos de interés con anotaciones
            if puntos_interes:
                for pt_x, label, color in puntos_interes:
                    try:
                        pt_x_val = float(pt_x)
                        # Usar math para punto único (más seguro para singularidades)
                        f_math = sp.lambdify(x, expr_sym, modules=['math'])
                        pt_y_val = f_math(pt_x_val)
                        
                        # Marcador
                        ax.scatter([pt_x_val], [pt_y_val], color=color, s=80, edgecolors='white', linewidth=1.5, zorder=5, label=label)
                        
                        # Líneas de proyección punteadas (Dashed)
                        ax.plot([pt_x_val, pt_x_val], [0, pt_y_val], color=color, linestyle='--', linewidth=1, alpha=0.7)
                        
                        # Anotación de texto con coordenadas
                        coord_text = f"{label}\n({pt_x_val:.2f}, {pt_y_val:.2f})"
                        offset_y = 15 if pt_y_val >= 0 else -30
                        
                        ax.annotate(coord_text, 
                                    xy=(pt_x_val, pt_y_val), 
                                    xytext=(0, offset_y),
                                    textcoords='offset points', 
                                    ha='center', 
                                    fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
                                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.6))
                                    
                    except Exception: pass

            # 5. Detalles finales
            ax.set_title("Visualización de la Función", fontsize=14, pad=20, fontweight='bold')
            ax.set_xlabel("x", loc='right', fontsize=12, style='italic')
            ax.set_ylabel("f(x)", loc='top', fontsize=12, style='italic', rotation=0)
            
            # Rejilla sutil
            ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
            
            # Leyenda mejorada
            ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9, shadow=True)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error al graficar:\n{e}", 
                    ha='center', va='center', transform=ax.transAxes, color='red')
        
        self.figure.tight_layout()
        self.canvas.draw()

    def _graficar(self, expr_str, x_range, puntos_interes):
        ax = self.figure.add_subplot(111)
        
        # Limpieza básica de la expresión para compatibilidad
        # Convertimos ^ a ** y e^ a exp() para que Sympy/Python lo entiendan
        expr_clean = expr_str.replace("^", "**").replace("e^", "exp")
        
        try:
            # Usar sympy para convertir el string a una función numérica vectorizada (para numpy)
            expr_sym = sp.sympify(expr_clean)
            f_np = sp.lambdify(x, expr_sym, modules=['numpy'])
            
            # Definir el rango de graficación (eje X)
            x_min, x_max = x_range
            # Añadir un pequeño margen si los puntos son iguales o muy cercanos
            if abs(x_max - x_min) < 1e-9:
                margin = 1.0
            else:
                margin = (x_max - x_min) * 0.2
            
            t = np.linspace(x_min - margin, x_max + margin, 400)
            
            # Evaluar la función en el rango
            y = f_np(t)

            # Graficar la curva principal
            ax.plot(t, y, label=f"f(x)={expr_str}", color="#2E86C1", linewidth=2)
            
            # Dibujar ejes X e Y
            ax.axhline(0, color='black', linewidth=0.8)
            ax.axvline(0, color='black', linewidth=0.8)

            # Graficar puntos de interés (raíces, extremos, etc.)
            if puntos_interes:
                for pt_x, label, color in puntos_interes:
                    try:
                        # Convertir a float por seguridad
                        pt_x_val = float(pt_x)
                        # Evaluar función en ese punto específico
                        # Usamos la versión sympy -> math para un solo punto por seguridad
                        f_math = sp.lambdify(x, expr_sym, modules=['math'])
                        pt_y_val = f_math(pt_x_val)
                        
                        # Dibujar el punto
                        ax.plot(pt_x_val, pt_y_val, 'o', color=color, label=label, markersize=6)
                        
                        # Línea punteada vertical hacia el eje X
                        ax.vlines(pt_x_val, 0, pt_y_val, colors=color, linestyles='dashed', alpha=0.5)
                    except Exception:
                        pass # Ignorar puntos que no se puedan evaluar (ej. fuera de dominio)

            # Configuración final del gráfico
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title(f"Visualización de f(x)")
            
        except Exception as e:
            # Mostrar error en el lienzo si la función es inválida
            ax.text(0.5, 0.5, f"Error al graficar:\n{e}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red', fontsize=12)
        
        self.canvas.draw()


# ==================================================
# Clase principal de la pestaña
# ==================================================
class TabMetodosNumericos(QtWidgets.QWidget):
    """
    Pestaña que agrupa:
    1. Métodos de Raíces (Cerrados: Bisección, Regla Falsa)
    2. Métodos de Raíces (Abiertos: Newton-Raphson, Secante)
    3. Teoría de Errores y Propagación
    4. Notación Posicional
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Variables para almacenar los resultados y poder generar reportes o gráficas
        self.info_raices_cache = None
        self.info_newton_cache = None
        self.info_secante_cache = None

        main = QtWidgets.QVBoxLayout(self)

        # Contenedor de sub-pestañas
        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs)

        # ==================================================
        # Sub-pestaña 1: Métodos Cerrados
        # ==================================================
        tab_raices = QtWidgets.QWidget()
        lay_raices = QtWidgets.QVBoxLayout(tab_raices)

        grp_raices = QtWidgets.QGroupBox("Métodos Cerrados (Bisección / Regla Falsa)")
        v_raices = QtWidgets.QVBoxLayout(grp_raices)

        fila_fx = QtWidgets.QHBoxLayout()
        self.ed_expr = QtWidgets.QLineEdit("x^3 - x - 1")
        fila_fx.addWidget(QtWidgets.QLabel("f(x) ="))
        fila_fx.addWidget(self.ed_expr)
        v_raices.addLayout(fila_fx)

        fila_int = QtWidgets.QHBoxLayout()
        self.ed_a = QtWidgets.QLineEdit("1")
        self.ed_b = QtWidgets.QLineEdit("2")
        self.ed_tol = QtWidgets.QLineEdit("1e-4")
        self.ed_max = QtWidgets.QLineEdit("50")

        fila_int.addWidget(QtWidgets.QLabel("a:"))
        fila_int.addWidget(self.ed_a)
        fila_int.addWidget(QtWidgets.QLabel("b:"))
        fila_int.addWidget(self.ed_b)
        fila_int.addWidget(QtWidgets.QLabel("Tol:"))
        fila_int.addWidget(self.ed_tol)
        fila_int.addWidget(QtWidgets.QLabel("Iter:"))
        fila_int.addWidget(self.ed_max)
        v_raices.addLayout(fila_int)

        fila_met = QtWidgets.QHBoxLayout()
        self.cbo_metodo = QtWidgets.QComboBox()
        self.cbo_metodo.addItems(["Bisección", "Regla falsa"])
        fila_met.addWidget(QtWidgets.QLabel("Método:"))
        fila_met.addWidget(self.cbo_metodo)
        fila_met.addStretch(1)
        v_raices.addLayout(fila_met)

        # Botones Tab 1
        h_btns_raices = QtWidgets.QHBoxLayout()
        self.btn_calcular_raiz = btn("Calcular")
        self.btn_calcular_raiz.clicked.connect(self.on_calcular_raiz)
        
        self.btn_paso_a_paso = btn("Ver Pasos", kind="ghost")
        self.btn_paso_a_paso.clicked.connect(self.on_mostrar_detalles)
        self.btn_paso_a_paso.setEnabled(False)

        self.btn_graf_raices = btn("Graficar", kind="ghost")
        self.btn_graf_raices.clicked.connect(self.on_graficar_raices)

        h_btns_raices.addWidget(self.btn_calcular_raiz)
        h_btns_raices.addWidget(self.btn_paso_a_paso)
        h_btns_raices.addWidget(self.btn_graf_raices)
        v_raices.addLayout(h_btns_raices)

        self.out_raices = OutputArea()
        v_raices.addWidget(self.out_raices)

        lay_raices.addWidget(grp_raices)
        lay_raices.addStretch(1)
        self.tabs.addTab(tab_raices, "Cerrados")

        # ==================================================
        # Sub-pestaña 2: Newton-Raphson
        # ==================================================
        tab_newton = QtWidgets.QWidget()
        lay_newton = QtWidgets.QVBoxLayout(tab_newton)

        grp_newton = QtWidgets.QGroupBox("Newton-Raphson (Método Abierto)")
        v_newton = QtWidgets.QVBoxLayout(grp_newton)

        fila_fx_n = QtWidgets.QHBoxLayout()
        self.ed_expr_newton = QtWidgets.QLineEdit("x^3 - x - 1")
        fila_fx_n.addWidget(QtWidgets.QLabel("f(x) ="))
        fila_fx_n.addWidget(self.ed_expr_newton)
        v_newton.addLayout(fila_fx_n)

        fila_newton = QtWidgets.QHBoxLayout()
        self.ed_x0_newton = QtWidgets.QLineEdit("1.5")
        self.ed_tol_newton = QtWidgets.QLineEdit("1e-4")
        self.ed_max_newton = QtWidgets.QLineEdit("50")

        fila_newton.addWidget(QtWidgets.QLabel("x0:"))
        fila_newton.addWidget(self.ed_x0_newton)
        fila_newton.addWidget(QtWidgets.QLabel("Tol:"))
        fila_newton.addWidget(self.ed_tol_newton)
        fila_newton.addWidget(QtWidgets.QLabel("Iter:"))
        fila_newton.addWidget(self.ed_max_newton)
        v_newton.addLayout(fila_newton)

        # Botones Tab 2
        h_btns_newton = QtWidgets.QHBoxLayout()
        self.btn_newton = btn("Calcular")
        self.btn_newton.clicked.connect(self.on_calcular_newton)
        
        self.btn_paso_newton = btn("Ver Pasos", kind="ghost")
        self.btn_paso_newton.clicked.connect(self.on_mostrar_detalles_newton)
        self.btn_paso_newton.setEnabled(False)

        self.btn_graf_newton = btn("Graficar", kind="ghost")
        self.btn_graf_newton.clicked.connect(self.on_graficar_newton)

        h_btns_newton.addWidget(self.btn_newton)
        h_btns_newton.addWidget(self.btn_paso_newton)
        h_btns_newton.addWidget(self.btn_graf_newton)
        v_newton.addLayout(h_btns_newton)

        self.out_newton = OutputArea()
        v_newton.addWidget(self.out_newton)

        lay_newton.addWidget(grp_newton)
        lay_newton.addStretch(1)
        self.tabs.addTab(tab_newton, "Newton")

        # ==================================================
        # Sub-pestaña 3: Secante
        # ==================================================
        tab_secante = QtWidgets.QWidget()
        lay_secante = QtWidgets.QVBoxLayout(tab_secante)

        grp_sec = QtWidgets.QGroupBox("Secante (Método Abierto)")
        v_sec = QtWidgets.QVBoxLayout(grp_sec)

        fila_fx_s = QtWidgets.QHBoxLayout()
        self.ed_expr_secante = QtWidgets.QLineEdit("x^3 - x - 1")
        fila_fx_s.addWidget(QtWidgets.QLabel("f(x) ="))
        fila_fx_s.addWidget(self.ed_expr_secante)
        v_sec.addLayout(fila_fx_s)

        fila_sec1 = QtWidgets.QHBoxLayout()
        self.ed_x0_secante = QtWidgets.QLineEdit("1")
        self.ed_x1_secante = QtWidgets.QLineEdit("2")
        fila_sec1.addWidget(QtWidgets.QLabel("x0:"))
        fila_sec1.addWidget(self.ed_x0_secante)
        fila_sec1.addSpacing(10)
        fila_sec1.addWidget(QtWidgets.QLabel("x1:"))
        fila_sec1.addWidget(self.ed_x1_secante)
        v_sec.addLayout(fila_sec1)

        fila_sec2 = QtWidgets.QHBoxLayout()
        self.ed_tol_secante = QtWidgets.QLineEdit("1e-4")
        self.ed_max_secante = QtWidgets.QLineEdit("50")
        fila_sec2.addWidget(QtWidgets.QLabel("Tol:"))
        fila_sec2.addWidget(self.ed_tol_secante)
        fila_sec2.addWidget(QtWidgets.QLabel("Iter:"))
        fila_sec2.addWidget(self.ed_max_secante)
        v_sec.addLayout(fila_sec2)

        # Botones Tab 3
        h_btns_sec = QtWidgets.QHBoxLayout()
        self.btn_secante = btn("Calcular")
        self.btn_secante.clicked.connect(self.on_calcular_secante)

        self.btn_paso_secante = btn("Ver Pasos", kind="ghost")
        self.btn_paso_secante.clicked.connect(self.on_mostrar_detalles_secante)
        self.btn_paso_secante.setEnabled(False)

        self.btn_graf_secante = btn("Graficar", kind="ghost")
        self.btn_graf_secante.clicked.connect(self.on_graficar_secante)

        h_btns_sec.addWidget(self.btn_secante)
        h_btns_sec.addWidget(self.btn_paso_secante)
        h_btns_sec.addWidget(self.btn_graf_secante)
        v_sec.addLayout(h_btns_sec)

        self.out_secante = OutputArea()
        v_sec.addWidget(self.out_secante)

        lay_secante.addWidget(grp_sec)
        lay_secante.addStretch(1)
        self.tabs.addTab(tab_secante, "Secante")

        # ==================================================
        # Sub-pestaña 4: Errores y Propagación
        # ==================================================
        tab_errores = QtWidgets.QWidget()
        lay_errores = QtWidgets.QVBoxLayout(tab_errores)

        grp_err = QtWidgets.QGroupBox("Errores Numéricos")
        v_err = QtWidgets.QVBoxLayout(grp_err)

        fila_val = QtWidgets.QHBoxLayout()
        self.ed_val_real = QtWidgets.QLineEdit("3.1416")
        self.ed_val_aprox = QtWidgets.QLineEdit("3.14")
        fila_val.addWidget(QtWidgets.QLabel("Valor real m:"))
        fila_val.addWidget(self.ed_val_real)
        fila_val.addSpacing(10)
        fila_val.addWidget(QtWidgets.QLabel("Valor aproximado x:"))
        fila_val.addWidget(self.ed_val_aprox)
        v_err.addLayout(fila_val)

        fila_btn_err = QtWidgets.QHBoxLayout()
        self.btn_calc_err = btn("Calcular errores")
        self.btn_calc_err.clicked.connect(self.on_calcular_errores)
        self.btn_tipos_err = btn("Tipos de error (teoría)", kind="ghost")
        self.btn_tipos_err.clicked.connect(self.on_tipos_error)
        self.btn_ej_flot = btn("Ejemplo flotante", kind="ghost")
        self.btn_ej_flot.clicked.connect(self.on_ejemplo_flotante)
        
        fila_btn_err.addWidget(self.btn_calc_err)
        fila_btn_err.addWidget(self.btn_tipos_err)
        fila_btn_err.addWidget(self.btn_ej_flot)
        fila_btn_err.addStretch(1)
        v_err.addLayout(fila_btn_err)

        self.out_err = OutputArea()
        v_err.addWidget(self.out_err)
        lay_errores.addWidget(grp_err)

        grp_prop = QtWidgets.QGroupBox("Propagación del Error")
        v_prop = QtWidgets.QVBoxLayout(grp_prop)

        fila_fx_prop = QtWidgets.QHBoxLayout()
        self.ed_fx_prop = QtWidgets.QLineEdit("x^2 + 3x")
        fila_fx_prop.addWidget(QtWidgets.QLabel("f(x) ="))
        fila_fx_prop.addWidget(self.ed_fx_prop)
        v_prop.addLayout(fila_fx_prop)

        fila_x0 = QtWidgets.QHBoxLayout()
        self.ed_x0 = QtWidgets.QLineEdit("2")
        self.ed_dx = QtWidgets.QLineEdit("0.01")
        fila_x0.addWidget(QtWidgets.QLabel("Valor central x0:"))
        fila_x0.addWidget(self.ed_x0)
        fila_x0.addSpacing(10)
        fila_x0.addWidget(QtWidgets.QLabel("Error en x (Δx):"))
        fila_x0.addWidget(self.ed_dx)
        v_prop.addLayout(fila_x0)

        self.btn_prop = btn("Calcular propagación")
        self.btn_prop.clicked.connect(self.on_propagacion)
        v_prop.addWidget(self.btn_prop)

        self.out_prop = OutputArea()
        v_prop.addWidget(self.out_prop)

        lay_errores.addWidget(grp_prop)
        lay_errores.addStretch(1)
        self.tabs.addTab(tab_errores, "Errores")

        # ==================================================
        # Sub-pestaña 5: Notación Posicional
        # ==================================================
        tab_notacion = QtWidgets.QWidget()
        lay_not = QtWidgets.QVBoxLayout(tab_notacion)

        grp_not = QtWidgets.QGroupBox("Notación posicional (base 10 y base 2)")
        v_not = QtWidgets.QVBoxLayout(grp_not)

        fila_n = QtWidgets.QHBoxLayout()
        self.ed_entero = QtWidgets.QLineEdit("472")
        fila_n.addWidget(QtWidgets.QLabel("Número entero:"))
        fila_n.addWidget(self.ed_entero)
        v_not.addLayout(fila_n)

        fila_btn_not = QtWidgets.QHBoxLayout()
        self.btn_base10 = btn("Base 10")
        self.btn_base10.clicked.connect(self.on_base10)
        self.btn_base2 = btn("Base 2")
        self.btn_base2.clicked.connect(self.on_base2)
        fila_btn_not.addWidget(self.btn_base10)
        fila_btn_not.addWidget(self.btn_base2)
        fila_btn_not.addStretch(1)
        v_not.addLayout(fila_btn_not)

        self.out_not = OutputArea()
        v_not.addWidget(self.out_not)

        lay_not.addWidget(grp_not)
        lay_not.addStretch(1)
        self.tabs.addTab(tab_notacion, "Notación")

    # ==================================================
    # LÓGICA DE GRAFICADO
    # ==================================================
    def on_graficar_raices(self):
        """Grafica f(x) en el intervalo [a, b] para métodos cerrados."""
        try:
            expr = self.ed_expr.text().strip()
            a = float(self.ed_a.text())
            b = float(self.ed_b.text())
            
            # Puntos a marcar: extremos a y b
            puntos = [(a, "a", "red"), (b, "b", "red")]
            
            # Si ya hay un cálculo realizado, agregar la raíz encontrada
            if self.info_raices_cache and self.info_raices_cache.get("raiz_aproximada"):
                xr = self.info_raices_cache["raiz_aproximada"]
                puntos.append((xr, "Raiz", "green"))

            # Crear ventana gráfica
            # Ampliamos un poco el rango para ver contexto
            margen = abs(b - a) * 0.1
            VentanaGrafica(expr, (min(a,b)-margen, max(a,b)+margen), puntos, self).exec()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error al graficar", f"Verifique los datos de entrada.\n\nDetalle: {e}")

    def on_graficar_newton(self):
        """Grafica f(x) alrededor de x0 para Newton."""
        try:
            expr = self.ed_expr_newton.text().strip()
            x0 = float(self.ed_x0_newton.text())
            
            # Puntos iniciales
            puntos = [(x0, "x0", "red")]
            
            # Definir rango por defecto
            x_min, x_max = x0 - 2, x0 + 2

            # Si hay resultado, ajustar rango y marcar raíz
            if self.info_newton_cache and self.info_newton_cache.get("raiz_aproximada"):
                xr = self.info_newton_cache["raiz_aproximada"]
                puntos.append((xr, "Raiz", "green"))
                
                # Ajustar ventana para incluir x0 y la raíz
                mn = min(x0, xr)
                mx = max(x0, xr)
                dist = mx - mn
                if dist < 1e-9: dist = 1.0
                margin = dist * 0.5
                x_min, x_max = mn - margin, mx + margin

            VentanaGrafica(expr, (x_min, x_max), puntos, self).exec()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error al graficar", f"Verifique los datos de entrada.\n\nDetalle: {e}")

    def on_graficar_secante(self):
        """Grafica f(x) alrededor de x0 y x1 para Secante."""
        try:
            expr = self.ed_expr_secante.text().strip()
            x0 = float(self.ed_x0_secante.text())
            x1 = float(self.ed_x1_secante.text())
            
            puntos = [(x0, "x0", "red"), (x1, "x1", "orange")]
            
            mn = min(x0, x1)
            mx = max(x0, x1)

            if self.info_secante_cache and self.info_secante_cache.get("raiz_aproximada"):
                xr = self.info_secante_cache["raiz_aproximada"]
                puntos.append((xr, "Raiz", "green"))
                mn = min(mn, xr)
                mx = max(mx, xr)

            dist = mx - mn
            if dist < 1e-9: dist = 1.0
            margin = dist * 0.5
            
            VentanaGrafica(expr, (mn - margin, mx + margin), puntos, self).exec()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error al graficar", f"Verifique los datos de entrada.\n\nDetalle: {e}")

    # ==================================================
    # Métodos Auxiliares
    # ==================================================
    def _tabla_pasos(self, info):
        """
        Formatea la tabla de iteraciones para Bisección / Regla Falsa /
        Newton-Raphson / Secante.
        """
        pasos = info.get("pasos", [])
        if not pasos:
            return "No se generaron iteraciones."

        # Ajustar cabecera según método
        metodo = info.get("metodo", "")
        if metodo == "Newton-Raphson":
             header = (
                "Iter   xi         f(xi)        f'(xi)       ea           er%\n"
                "-------------------------------------------------------------------"
            )
        elif metodo == "Secante":
             header = (
                "Iter   xi_ant     xi         f(xi)        ea           er%\n"
                "-------------------------------------------------------------------"
            )
        else:
            header = (
                "Iter   a          b          xr         f(xr)        ea           er%\n"
                "-------------------------------------------------------------------------------"
            )
        filas = [header]

        for p in pasos:
            it = p["numero_de_iteracion"]
            
            # Helper para formatear valores o nulos
            def fval(v):
                if v is None: return "   ---   "
                return fmt_number(float(v), DEFAULT_DEC, False).rjust(10)

            ea = fval(p.get("error_absoluto_ea"))
            erp = fval(p.get("error_relativo_porcentual"))

            if metodo == "Newton-Raphson":
                xi = fval(p.get("xi"))
                fxi = fval(p.get("f_xi"))
                dfxi = fval(p.get("df_xi"))
                fila = f"{it:>3}  {xi} {fxi} {dfxi} {ea} {erp}"
            elif metodo == "Secante":
                xant = fval(p.get("xi_ant"))
                xi = fval(p.get("xi"))
                fxi = fval(p.get("f_xi"))
                fila = f"{it:>3}  {xant} {xi} {fxi} {ea} {erp}"
            else:
                a = fval(p.get("extremo_izquierdo_a"))
                b = fval(p.get("extremo_derecho_b"))
                xr = fval(p.get("aproximacion_actual_xr"))
                fxr = fval(p.get("valor_de_la_funcion_en_xr"))
                fila = f"{it:>3}  {a} {b} {xr} {fxr} {ea} {erp}"
            
            filas.append(fila)

        return "\n".join(filas)

    def _mostrar_dialogo(self, texto, titulo):
        try:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f"Paso a Paso Detallado - {titulo}")
            dialog.resize(700, 700)
            layout = QtWidgets.QVBoxLayout(dialog)
            
            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Consolas", 10))
            text_edit.setPlainText(texto)
            
            layout.addWidget(text_edit)
            
            btn_ok = QtWidgets.QPushButton("Cerrar")
            btn_ok.clicked.connect(dialog.accept)
            layout.addWidget(btn_ok)
            
            dialog.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al generar reporte:\n{e}")

    # ==================================================
    # Handlers de Cálculo
    # ==================================================
    def on_calcular_raiz(self):
        try:
            expr = self.ed_expr.text().strip()
            a = float(self.ed_a.text())
            b = float(self.ed_b.text())
            tol = float(self.ed_tol.text())
            max_iter = int(self.ed_max.text())

            if self.cbo_metodo.currentText().startswith("Bisección"):
                info = biseccion_descriptiva(expr, a, b, tol, max_iter)
            else:
                info = regla_falsa_descriptiva(expr, a, b, tol, max_iter)

            self.info_raices_cache = info
            self.btn_paso_a_paso.setEnabled(True)

            # Construir salida principal
            lineas = []
            lineas.append(f"Método: {info['metodo']}")
            lineas.append(f"f(x) = {expr}")
            lineas.append(f"Intervalo inicial: [{a}, {b}]")
            lineas.append(f"Tolerancia: {tol}")
            lineas.append("")
            lineas.append(self._tabla_pasos(info))
            lineas.append("")

            lineas.append("Raíz aproximada: " + fmt_number(info["raiz_aproximada"], DEFAULT_DEC, False))
            lineas.append("f(raíz) ≈ " + fmt_number(info["valor_funcion_en_raiz"], DEFAULT_DEC, False))

            pasos = info.get("pasos", [])
            if pasos:
                ultimo = pasos[-1]
                iter_tot = info.get("iteraciones_totales", len(pasos))
                erp_final = ultimo.get("error_relativo_porcentual")
                ea_final = ultimo.get("error_absoluto_ea")

                lineas.append("Iteraciones totales: " + str(iter_tot))
                erp_str = "---" if erp_final is None else fmt_number(erp_final, DEFAULT_DEC, False)
                lineas.append("Error relativo porcentual final er% = " + erp_str)

                criterio = info.get("criterio_de_paro")
                if criterio == "tolerancia" and ea_final is not None and ea_final <= tol:
                    lineas.append("Comentario: Convergencia alcanzada según tolerancia.")
                elif criterio == "max_iter":
                    lineas.append("Comentario: Se alcanzó el máximo de iteraciones.")
                else:
                    lineas.append(f"Comentario: Parada por {criterio}.")

            lineas.append("")
            lineas.append("Fórmulas usadas: ea = |xr - x_ant|, er = ea/|xr|.")

            self.out_raices.clear_and_write("\n".join(lineas))
        except Exception as e:
            msg = str(e)
            if "El intervalo no es válido" in msg:
                self.out_raices.clear_and_write("Error: El intervalo no es válido (f(a)*f(b) > 0).")
            else:
                self.out_raices.clear_and_write(f"Error: {msg}")

    def on_mostrar_detalles(self):
        if not self.info_raices_cache: return
        self._mostrar_dialogo(generar_reporte_paso_a_paso(self.info_raices_cache), self.info_raices_cache["metodo"])

    def on_calcular_newton(self):
        try:
            expr = self.ed_expr_newton.text().strip()
            x0 = float(self.ed_x0_newton.text())
            tol = float(self.ed_tol_newton.text())
            max_iter = int(self.ed_max_newton.text())

            info = newton_raphson_descriptiva(expr, x0, tol, max_iter)
            self.info_newton_cache = info
            self.btn_paso_newton.setEnabled(True)

            lineas = []
            lineas.append(f"Método: {info['metodo']}")
            lineas.append(f"f(x) = {expr}")
            lineas.append(f"x0 inicial: {x0}")
            lineas.append("")
            lineas.append(self._tabla_pasos(info))
            lineas.append("")
            lineas.append("Raíz aproximada: " + fmt_number(info["raiz_aproximada"], DEFAULT_DEC, False))
            
            pasos = info.get("pasos", [])
            if pasos:
                ultimo = pasos[-1]
                iter_tot = info.get("iteraciones_totales")
                erp_final = ultimo.get("error_relativo_porcentual")
                lineas.append("Iteraciones totales: " + str(iter_tot))
                erp_str = "---" if erp_final is None else fmt_number(erp_final, DEFAULT_DEC, False)
                lineas.append("Error relativo porcentual final er% = " + erp_str)

            self.out_newton.clear_and_write("\n".join(lineas))
        except Exception as e:
            self.out_newton.clear_and_write(f"Error: {e}")

    def on_mostrar_detalles_newton(self):
        if not self.info_newton_cache: return
        self._mostrar_dialogo(generar_reporte_abierto(self.info_newton_cache), "Newton-Raphson")

    def on_calcular_secante(self):
        try:
            expr = self.ed_expr_secante.text().strip()
            x0 = float(self.ed_x0_secante.text())
            x1 = float(self.ed_x1_secante.text())
            tol = float(self.ed_tol_secante.text())
            max_iter = int(self.ed_max_secante.text())

            info = secante_descriptiva(expr, x0, x1, tol, max_iter)
            self.info_secante_cache = info
            self.btn_paso_secante.setEnabled(True)

            lineas = []
            lineas.append(f"Método: {info['metodo']}")
            lineas.append(f"f(x) = {expr}")
            lineas.append(f"Puntos iniciales: x0={x0}, x1={x1}")
            lineas.append("")
            lineas.append(self._tabla_pasos(info))
            lineas.append("")
            lineas.append("Raíz aproximada: " + fmt_number(info["raiz_aproximada"], DEFAULT_DEC, False))
            
            pasos = info.get("pasos", [])
            if pasos:
                ultimo = pasos[-1]
                iter_tot = info.get("iteraciones_totales")
                erp_final = ultimo.get("error_relativo_porcentual")
                lineas.append("Iteraciones totales: " + str(iter_tot))
                erp_str = "---" if erp_final is None else fmt_number(erp_final, DEFAULT_DEC, False)
                lineas.append("Error relativo porcentual final er% = " + erp_str)

            self.out_secante.clear_and_write("\n".join(lineas))
        except Exception as e:
            self.out_secante.clear_and_write(f"Error: {e}")

    def on_mostrar_detalles_secante(self):
        if not self.info_secante_cache: return
        self._mostrar_dialogo(generar_reporte_abierto(self.info_secante_cache), "Secante")

    def on_calcular_errores(self):
        try:
            m = float(self.ed_val_real.text())
            x_val = float(self.ed_val_aprox.text())
            info = calcular_errores(m, x_val)
            
            dec_err = 6
            ea = fmt_number(info["error_absoluto"], dec_err, False)
            er = "---" if info["error_relativo"] is None else fmt_number(info["error_relativo"], dec_err, False)
            erp = "---" if info["error_relativo_porcentual"] is None else fmt_number(info["error_relativo_porcentual"], dec_err, False)

            texto = (
                f"Valor real m = {info['valor_real']}\n"
                f"Valor aproximado x = {info['valor_aproximado']}\n\n"
                f"Error absoluto ea = |m - x| = {ea}\n"
                f"Error relativo er = ea / |m| = {er}\n"
                f"Error relativo porcentual er% = er × 100 = {erp}"
            )
            self.out_err.clear_and_write(texto)
        except Exception as e:
            self.out_err.clear_and_write(f"Error: {e}")

    def on_tipos_error(self):
        self.out_err.clear_and_write(tipos_de_error_texto())

    def on_ejemplo_flotante(self):
        self.out_err.clear_and_write(ejemplo_punto_flotante_texto())

    def on_base10(self):
        try:
            n = int(self.ed_entero.text())
            self.out_not.clear_and_write(descomponer_base10(n))
        except Exception as e:
            self.out_not.clear_and_write(f"Error: {e}")

    def on_base2(self):
        try:
            n = int(self.ed_entero.text())
            self.out_not.clear_and_write(descomponer_base2(n))
        except Exception as e:
            self.out_not.clear_and_write(f"Error: {e}")

    def on_propagacion(self):
        try:
            expr = self.ed_fx_prop.text().strip()
            x0 = float(self.ed_x0.text())
            dx = float(self.ed_dx.text())

            info = propagacion_error(expr, x0, dx)

            texto = (
                f"f(x) = {expr}\n"
                f"x0 = {info['x0']}, Δx = {info['delta_x']}\n\n"
                f"f(x0) = {fmt_number(info['y0'], DEFAULT_DEC, False)}\n"
                f"f(x0 + Δx) = {fmt_number(info['y1'], DEFAULT_DEC, False)}\n"
                f"Δy ≈ f(x0 + Δx) − f(x0) = {fmt_number(info['delta_y'], DEFAULT_DEC, False)}\n\n"
                f"Error absoluto en y: {fmt_number(info['error_absoluto_y'], DEFAULT_DEC, False)}\n"
                f"Error relativo en y: "
                f"{'---' if info['error_relativo_y'] is None else fmt_number(info['error_relativo_y'], DEFAULT_DEC, False)}\n"
                f"Error relativo porcentual en y: "
                f"{'---' if info['error_relativo_porcentual_y'] is None else fmt_number(info['error_relativo_porcentual_y'], DEFAULT_DEC, False)}"
            )
            self.out_prop.clear_and_write(texto)
        except Exception as e:
            self.out_prop.clear_and_write(f"Error: {e}")