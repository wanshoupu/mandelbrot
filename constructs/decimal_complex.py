from dataclasses import dataclass
from decimal import Decimal

import numpy as np


@dataclass(frozen=True)
class DComplex:
    real: Decimal
    imag: Decimal

    def __abs__(self):
        return (self.real ** 2 + self.imag ** 2).sqrt()

    def __add__(self, other):
        if not isinstance(other, DComplex):
            other = DComplex.cast(other)
        return DComplex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        if not isinstance(other, DComplex):
            other = DComplex.cast(other)
        a, b = self.real, self.imag
        c, d = other.real, other.imag
        return DComplex(a * c - b * d, a * d + b * c)

    @classmethod
    def cast(cls, other) -> 'DComplex':
        return DComplex(Decimal(np.real(other)), Decimal(np.imag(other)))


def dcomplex_zeroes(shape):
    zero = DComplex(Decimal(0), Decimal(0))  # avoid Decimal(0) to preserve precision if needed
    arr = np.full(shape, zero, dtype=object)
    return arr


def dcomplex_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Shapes must match for addition.")
    return np.vectorize(lambda x, y: x + y, otypes=[object])(a, b)


def dcomplex_sq(a: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda x: x * x, otypes=[object])(a)


def dcomplex_abs(a: np.ndarray) -> np.ndarray:
    return np.vectorize(abs, otypes=[object])(a)
