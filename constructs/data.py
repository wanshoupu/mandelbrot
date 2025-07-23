import glob
import os
from dataclasses import dataclass, astuple
from multiprocessing import Pool

import numpy as np

FILE_PREFIX = 'tmp'
PIXEL_X, PIXEL_Y = 2560, 1600
CPU_CORES = 8
PARALLELISM = CPU_CORES * 2
THRESHOLD = 2


@dataclass
class PlotSpecs:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    iterations: int = None

    def __post_init__(self):
        if self.iterations is None:
            self.iterations = iter_heuristic(self)
        else:
            self.iterations = int(self.iterations)


@dataclass
class MandelbrotData:
    dataset: np.ndarray
    interior: np.ndarray
    rect: np.ndarray


def mandelbrot_dataset(specs: PlotSpecs, width, height, iterations):
    C = clingrid(specs, width, height)
    chunks = np.array_split(C, PARALLELISM, axis=0)
    with Pool(processes=CPU_CORES) as pool:
        results = pool.starmap(mandelbrot_calc, [(c, iterations) for c in chunks])

    # Merge back along rows
    diverging_order_chunks, mask_interior_chunks = zip(*results)
    diverging_order = np.vstack(diverging_order_chunks)
    mask_interior = np.vstack(mask_interior_chunks)
    return diverging_order, mask_interior


def mandelbrot_calc(C, iterations):
    Z = np.zeros_like(C)
    mask_interior = np.full(C.shape, True, dtype=bool)  # mask for interior points
    diverging_order = np.zeros(C.shape)  # the number of iterations it takes to reach diverging point (> THRESHOLD)
    for i in range(iterations):
        Z[mask_interior] = Z[mask_interior] ** 2 + C[mask_interior]
        norm = np.abs(Z)
        diverged = norm > THRESHOLD
        mask = diverged & mask_interior
        diverging_order[mask] = i + 1 - np.log(np.log2(np.array(norm[mask], dtype=np.float64)))
        mask_interior[mask] = False
    return diverging_order, mask_interior


def clingrid(rect, width, height):
    x = np.linspace(rect.xmin, rect.xmax, width)
    y = np.linspace(rect.ymin, rect.ymax, height)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    return C


def data_gen(rect: PlotSpecs, regen=False) -> MandelbrotData:
    filename = f"{FILE_PREFIX}-{rect.iterations}-{rect.xmin}-{rect.xmax}-{rect.ymin}-{rect.ymax}.npz"
    if regen or not os.path.exists(filename):
        print(f"Generating data for:\n  Rect{astuple(rect)}")
        dataset, interior = mandelbrot_dataset(rect, PIXEL_X, PIXEL_Y, rect.iterations)
        np.savez(filename, dataset=dataset, interior=interior, rect=np.array(astuple(rect)))
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


def iter_heuristic(rect):
    dx = rect.xmax - rect.xmin
    dy = rect.ymax - rect.ymin
    iterations = int(150 * np.log10(dx * dy) - 209 * np.log10(dx * dy) + 327)
    return min(iterations, 2048)
