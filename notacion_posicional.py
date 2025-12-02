# notacion_posicional.py
from __future__ import annotations
from utilidad import formatear_superindice

def descomponer_base10(n: int) -> str:
    """
    Devuelve un texto que muestra la descomposición de un entero en base 10.
    Ejemplo: 472 = 4×10² + 7×10¹ + 2×10⁰
    """
    n = int(n)
    if n < 0:
        signo = "-"
        n_abs = -n
    else:
        signo = ""
        n_abs = n

    digitos = list(str(n_abs))
    k = len(digitos)
    partes = []
    for i, d in enumerate(digitos):
        exp = k - i - 1
        # Usamos formatear_superindice para convertir ^2 -> ²
        parte_fmt = formatear_superindice(f"{d}×10^{exp}")
        partes.append(parte_fmt)

    suma_potencias = " + ".join(partes)
    return (
        f"Número en base 10: {signo}{n_abs}\n"
        f"Descomposición posicional:\n"
        f"{signo}{n_abs} = {suma_potencias}"
    )

def descomponer_base2(n: int) -> str:
    """
    Devuelve un texto que muestra la descomposición de un entero en base 2.
    """
    n = int(n)
    if n < 0:
        signo = "-"
        n_abs = -n
    else:
        signo = ""
        n_abs = n

    bin_str = bin(n_abs)[2:] or "0"
    k = len(bin_str)
    partes = []
    for i, bit in enumerate(bin_str):
        exp = k - i - 1
        # Usamos formatear_superindice
        parte_fmt = formatear_superindice(f"{bit}×2^{exp}")
        partes.append(parte_fmt)

    suma_potencias = " + ".join(partes)
    return (
        f"Número en base 10: {signo}{n_abs}\n"
        f"Representación binaria: {signo}{bin_str}₂\n"
        f"Descomposición posicional:\n"
        f"{signo}{n_abs} = {suma_potencias}"
    )