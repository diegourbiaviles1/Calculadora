# core/secante.py

import sympy as sp

def calcular_secante(funcion_str, x0, x1, tolerancia=1e-5, iter_max=50):
    """
    Implementa el método de la Secante usando error aproximado absoluto
    |x_{i+1} - x_i| como criterio de paro, igual que en las diapositivas.

    Parámetros
    ----------
    funcion_str : str
        Función f(x) en formato cadena, por ejemplo: "x**3 - 5*x + 1".
    x0, x1 : float
        Dos valores iniciales.
    tolerancia : float
        Error aproximado absoluto máximo permitido.
    iter_max : int
        Número máximo de iteraciones.

    Retorna
    -------
    resultado : dict
        {
          "raiz": float,
          "iteraciones": int,
          "convergio": bool,
          "tabla": [ { "i": ..., "xi_1": ..., "xi": ..., "f_xi_1": ..., "f_xi": ..., "xi1": ..., "ea": ... }, ... ]
        }
    """

    x = sp.Symbol('x')
    f = sp.sympify(funcion_str)
    f_num = sp.lambdify(x, f, 'math')

    tabla = []

    x_im1 = float(x0)  # x_{i-1}
    x_i   = float(x1)  # x_i
    ea = None

    for i in range(1, iter_max + 1):
        f_x_im1 = f_num(x_im1)
        f_x_i   = f_num(x_i)

        denom = (f_x_i - f_x_im1)
        if denom == 0:
            tabla.append({
                "i": i,
                "xi_1": x_im1,
                "xi": x_i,
                "f_xi_1": f_x_im1,
                "f_xi": f_x_i,
                "xi1": None,
                "ea": None
            })
            return {
                "raiz": x_i,
                "iteraciones": i,
                "convergio": False,
                "tabla": tabla,
                "mensaje": "Denominador cero en la fórmula de la secante: no se puede continuar."
            }

        # Fórmula de la Secante:
        # x_{i+1} = x_i - f(x_i)*(x_i - x_{i-1}) / (f(x_i) - f(x_{i-1}))
        x_ip1 = x_i - f_x_i * (x_i - x_im1) / denom

        if i == 1:
            ea = None
        else:
            ea = abs(x_ip1 - x_i)

        tabla.append({
            "i": i,
            "xi_1": x_im1,
            "xi": x_i,
            "f_xi_1": f_x_im1,
            "f_xi": f_x_i,
            "xi1": x_ip1,
            "ea": ea
        })

        # Criterio de paro: error aproximado absoluto
        if ea is not None and ea < tolerancia:
            return {
                "raiz": x_ip1,
                "iteraciones": i,
                "convergio": True,
                "tabla": tabla,
                "mensaje": "Criterio de paro satisfecho: error aproximado absoluto < tolerancia."
            }

        # Actualizar para la siguiente iteración
        x_im1, x_i = x_i, x_ip1

    # Si se alcanzan las iteraciones máximas
    return {
        "raiz": x_i,
        "iteraciones": iter_max,
        "convergio": False,
        "tabla": tabla,
        "mensaje": "Se alcanzó el número máximo de iteraciones sin cumplir la tolerancia."
    }
