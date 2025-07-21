import glob
import os
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from constructs.decimal_complex import dcomplex_zeroes, dcomplex_add, dcomplex_sq, dclingrid

FILE_PREFIX = 'tmp'
PIXEL_X, PIXEL_Y = 2560, 1600
ITER_N = 100
CPU_CORES = 8
PARALLELISM = CPU_CORES * 2


@dataclass
class Rect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def to_array(self):
        array = np.array([self.xmin, self.xmax, self.ymin, self.ymax])
        return array


@dataclass
class MandelbrotData:
    dataset: np.ndarray
    interior: np.ndarray
    rect: np.ndarray


def mandelbrot_dataset(rect: Rect, width, height):
    precision = min(rect.xmax - rect.xmin, rect.ymax - rect.ymin)
    use_dcomplex = np.isclose(precision, 0)
    use_dcomplex = False

    C = dclingrid(rect, width, height) if use_dcomplex else clingrid(rect, width, height)
    chunks = np.array_split(C, PARALLELISM, axis=0)
    with Pool(processes=CPU_CORES) as pool:
        results = pool.map(mandelbrot_calc_dcomplex if use_dcomplex else mandelbrot_calc, chunks)

    # Merge back along rows
    diverging_order_chunks, mask_interior_chunks = zip(*results)
    diverging_order = np.vstack(diverging_order_chunks)
    mask_interior = np.vstack(mask_interior_chunks)
    return diverging_order, mask_interior


def mandelbrot_calc_dcomplex(C):
    Z = dcomplex_zeroes(C.shape)
    mask_interior = np.full(C.shape, True, dtype=bool)  # mask for interior points
    diverging_order = np.zeros(C.shape)  # the number of iterations it takes to reach diverging point (> 2)
    for i in range(ITER_N):
        Z[mask_interior] = dcomplex_add(dcomplex_sq(Z[mask_interior]), C[mask_interior])
        norm = np.abs(Z)
        diverged = norm > 2
        mask = diverged & mask_interior
        diverging_order[mask] = i + 1 - np.log(np.log2(np.array(norm[mask], dtype=np.float64)))
        mask_interior[mask] = False
    return diverging_order, mask_interior


def mandelbrot_calc(C):
    Z = np.zeros_like(C)
    mask_interior = np.full(C.shape, True, dtype=bool)  # mask for interior points
    diverging_order = np.zeros(C.shape)  # the number of iterations it takes to reach diverging point (> 2)
    for i in range(ITER_N):
        Z[mask_interior] = Z[mask_interior] ** 2 + C[mask_interior]
        norm = np.abs(Z)
        diverged = norm > 2
        mask = diverged & mask_interior
        diverging_order[mask] = i + 1 - np.log(np.log2(np.array(norm[mask], dtype=np.float64)))
        mask_interior[mask] = False
    return diverging_order, mask_interior


def clingrid(rect, width, height):
    x = np.linspace(rect.xmin, rect.xmax, width)
    y = np.linspace(rect.ymin, rect.ymax, height)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    return C


def data_gen(rect: Rect, regen=False) -> MandelbrotData:
    filename = f"{FILE_PREFIX}-{rect.xmin}-{rect.xmax}-{rect.ymin}-{rect.ymax}.npz"
    if regen or not os.path.exists(filename):
        dataset, interior = mandelbrot_dataset(rect, PIXEL_X, PIXEL_Y)
        np.savez(filename, dataset=dataset, interior=interior, rect=rect.to_array())
    return data_load(filename)


def data_load(filename: str) -> MandelbrotData:
    mandelbrot = np.load(filename)
    dataset = mandelbrot['dataset']
    interior = mandelbrot['interior']
    rect = np.array(mandelbrot['rect'])
    # Apply custom colormap for exterior
    return MandelbrotData(dataset, interior, rect)


def cache_cleanup():
    filename_pattern = f"{FILE_PREFIX}-*.npz"
    for filename in glob.glob(filename_pattern):
        os.remove(filename)
