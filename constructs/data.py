import os
from dataclasses import dataclass, astuple
from multiprocessing import Pool

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from constructs.cache import cache_manager

PIXEL_X, PIXEL_Y = 2560, 1600
CPU_CORES = 8
PARALLELISM = CPU_CORES * 2
THRESHOLD = 2
complex_type = np.complex128

CMAP_EXT = LinearSegmentedColormap.from_list(
    "electric", ["#000428", "#004e92", "#00d4ff", "#ffffff"], N=1024
)


def iter_heuristic(rect):
    dx = rect.xmax - rect.xmin
    dy = rect.ymax - rect.ymin
    iterations = int(150 * np.log10(dx * dy) - 209 * np.log10(dx * dy) + 327)
    return min(iterations, 2048)


@dataclass
class PlotSpecs:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    iterations: int = None
    width: int = PIXEL_X
    height: int = PIXEL_Y

    def __post_init__(self):
        if self.iterations is None:
            self.iterations = iter_heuristic(self)
        else:
            self.iterations = int(self.iterations)


@dataclass(frozen=True)
class MandelbrotViz:
    img: np.ndarray
    cmap: LinearSegmentedColormap
    vmin: float
    vmax: float
    specs: PlotSpecs


@dataclass(frozen=True)
class MandelbrotData:
    escapes: np.ndarray
    interior: np.ndarray
    rect: np.ndarray
    Z: np.ndarray = None

    def to_viz_data(self) -> MandelbrotViz:
        escapes = self.escapes
        interior = self.interior
        specs = PlotSpecs(*self.rect)
        pixelx, pixely = escapes.shape

        # Create image array
        img = np.zeros((pixelx, pixely, 3))
        # Normalize exterior values for coloring
        vmin, vmax = escapes.min(), escapes.max()
        norm_div = escapes / vmax
        # Apply colormap to diverged (exterior) points
        img[~interior] = CMAP_EXT(norm_div[~interior])[:, :3]  # drop alpha
        # Set interior (non-diverged) points to solid color (e.g., black or red)
        img[interior] = [0, 0, 0]  # deep red interior
        return MandelbrotViz(img, CMAP_EXT, vmin, vmax, specs)


def mandelbrot_calc(C: np.array, iterations, Z: np.array):
    mask_interior = np.full(C.shape, True, dtype=bool)  # mask for interior points
    diverging_order = np.zeros(C.shape)  # the number of iterations it takes to reach diverging point (> THRESHOLD)
    for i in range(iterations):
        Z[mask_interior] = Z[mask_interior] ** 2 + C[mask_interior]
        norm = np.abs(Z)
        diverged = norm > THRESHOLD
        mask = diverged & mask_interior
        diverging_order[mask] = i + 1 - np.log(np.log2(np.array(norm[mask], dtype=np.float64)))
        mask_interior[mask] = False
    return diverging_order, mask_interior, Z


def clingrid(rect):
    x = np.linspace(rect.xmin, rect.xmax, rect.width, dtype=complex_type)
    y = np.linspace(rect.ymin, rect.ymax, rect.height, dtype=complex_type)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    return C


def data_gen(specs: PlotSpecs, regen=False, Z: np.array = None, iterations_delta: int = None) -> MandelbrotData:
    """
    If Z is none, calculate the mandelbrot dataset ab initio
    Otherwise, calculate the mandelbrot dataset based on the given Z with additional iterations given by iterations_delta.
    """
    filename = cache_manager.get_filename(specs)
    if regen or not os.path.exists(filename):
        print(f"Generating data for:\n  PlotSpecs{astuple(specs)}")
        C = clingrid(specs)
        if Z is None:
            Z = np.zeros_like(C)
            iterations_delta = specs.iterations
        else:
            assert Z.shape == C.shape
            assert iterations_delta is not None and iterations_delta > 0

        C_chunks = np.array_split(C, PARALLELISM, axis=0)
        Z_chunks = np.array_split(Z, PARALLELISM, axis=0)
        with Pool(processes=CPU_CORES) as pool:
            results = pool.starmap(mandelbrot_calc, [(c, iterations_delta, z) for c, z in zip(C_chunks, Z_chunks)])

        # Merge back along rows
        diverging_order_chunks, mask_interior_chunks, Z_chunks = zip(*results)
        diverging_order = np.vstack(diverging_order_chunks)
        mask_interior = np.vstack(mask_interior_chunks)
        Z = np.vstack(Z_chunks)
        rect = np.array([specs.xmin, specs.xmax, specs.ymin, specs.ymax])
        dataset = MandelbrotData(diverging_order, mask_interior, rect, Z)
        np.savez(filename, escapes=dataset.escapes, interior=dataset.interior, rect=np.array(astuple(specs)), Z=dataset.Z)
    return data_load(filename)


def data_load(filename: str) -> MandelbrotData:
    mandelbrot = np.load(filename)
    dataset = mandelbrot['escapes']
    interior = mandelbrot['interior']
    rect = np.array(mandelbrot['rect'])
    Z_payload = mandelbrot['Z']
    # Apply custom colormap for exterior
    return MandelbrotData(dataset, interior, rect, Z_payload)


def cache_cleanup():
    cache_manager.cleanup()
