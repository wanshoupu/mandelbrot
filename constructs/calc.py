from dataclasses import astuple
from multiprocessing import Pool

import numpy as np

from constructs.cache import cache_manager
from constructs.model import PlotSpecs, MandelbrotData

CPU_CORES = 8
PARALLELISM = CPU_CORES * 2
THRESHOLD = 2
complex_type = np.complex128


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


def clingrid(specs: PlotSpecs):
    x = np.linspace(specs.xmin, specs.xmax, specs.width, dtype=complex_type)
    y = np.linspace(specs.ymin, specs.ymax, specs.height, dtype=complex_type)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    return C


def data_gen(specs: PlotSpecs, regen=False) -> MandelbrotData:
    """
    If Z is none, calculate the mandelbrot dataset ab initio
    Otherwise, calculate the mandelbrot dataset based on the given Z with additional iterations given by iterations_delta.
    """
    if regen or not cache_manager.exists(specs):
        print(f"Generating data for:\n  PlotSpecs{astuple(specs)}")
        C = clingrid(specs)
        closest_dataset = cache_manager.get_closest(specs)
        if closest_dataset is None:
            Z = np.zeros_like(C)
            iterations_delta = specs.iterations
        else:
            Z = closest_dataset.Z
            iterations_delta = specs.iterations - closest_dataset.to_specs().iterations

        dataset = data_regen(C, Z, iterations_delta, specs)
        cache_manager.commit(specs, dataset)
    return cache_manager.get(specs)


def data_regen(C: np.array, Z: np.array, iterations_delta: int, specs: PlotSpecs) -> MandelbrotData:
    C_chunks = np.array_split(C, PARALLELISM, axis=0)
    Z_chunks = np.array_split(Z, PARALLELISM, axis=0)
    with Pool(processes=CPU_CORES) as pool:
        results = pool.starmap(mandelbrot_calc, [(c, iterations_delta, z) for c, z in zip(C_chunks, Z_chunks)])
    # Merge back along rows
    diverging_order_chunks, mask_interior_chunks, Z_chunks = zip(*results)
    diverging_order = np.vstack(diverging_order_chunks)
    mask_interior = np.vstack(mask_interior_chunks)
    Z = np.vstack(Z_chunks)
    dataset = MandelbrotData(diverging_order, mask_interior, np.array(astuple(specs)), Z)
    return dataset
