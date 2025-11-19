from __future__ import annotations


def descomponer_base10(n: int) -> str:
    """
    Devuelve un texto que muestra la descomposición de un entero en base 10.
    Ejemplo:
        472 = 4×10^2 + 7×10^1 + 2×10^0
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
        partes.append(f"{d}×10^{exp}")

    suma_potencias = " + ".join(partes)
    return (
        f"Número en base 10: {signo}{n_abs}\n"
        f"Descomposición posicional:\n"
        f"{signo}{n_abs} = {suma_potencias}"
    )


def descomponer_base2(n: int) -> str:
    """
    Devuelve un texto que muestra la descomposición de un entero en base 2.
    Ejemplo:
        13 = (1101)_2 = 1×2^3 + 1×2^2 + 0×2^1 + 1×2^0
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
        partes.append(f"{bit}×2^{exp}")

    suma_potencias = " + ".join(partes)
    return (
        f"Número en base 10: {signo}{n_abs}\n"
        f"Representación binaria: {signo}{bin_str}_2\n"
        f"Descomposición posicional:\n"
        f"{signo}{n_abs} = {suma_potencias}"
    )