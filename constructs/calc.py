from dataclasses import astuple
from decimal import Decimal
from multiprocessing import Pool, Event

import numpy as np

from constructs.cache import cache_manager
from constructs.decimal_complex import dcomplex_zeroes, dcomplex_add, dcomplex_sq, DComplex, dcomplex_abs
from constructs.model import PlotSpecs, MandelbrotData

CPU_CORES = 8
PARALLELISM = CPU_CORES * 2
THRESHOLD = 2
complex_type = np.longcomplex
ATOL = 1e-16


def mandelbrot_calc_dcomplex(C: np.array, iterations, Z: np.array, cancel_event: Event):
    mask_interior = np.full(C.shape, True, dtype=bool)  # mask for interior points
    diverging_order = np.zeros(C.shape)  # the number of iterations it takes to reach diverging point (> 2)
    for i in range(iterations):
        if cancel_event is not None and cancel_event.is_set():
            break
        Z[mask_interior] = dcomplex_add(dcomplex_sq(Z[mask_interior]), C[mask_interior])
        norm = dcomplex_abs(Z)
        diverged = norm > 2
        mask = diverged & mask_interior
        diverging_order[mask] = i + 1 - np.log(np.log2(np.array(norm[mask], dtype=np.float64)))
        mask_interior[mask] = False
    return diverging_order, mask_interior


def mandelbrot_calc(C: np.array, iterations, Z: np.array, cancel_event: Event):
    mask_interior = np.full(C.shape, True, dtype=bool)  # mask for interior points
    diverging_order = np.zeros(C.shape)  # the number of iterations it takes to reach diverging point (> THRESHOLD)
    for i in range(iterations):
        if cancel_event is not None and cancel_event.is_set():
            break
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


def data_gen(specs: PlotSpecs, regen=False, cancel_event: Event = None) -> MandelbrotData:
    """
    If Z is none, calculate the mandelbrot dataset ab initio
    Otherwise, calculate the mandelbrot dataset based on the given Z with additional iterations given by iterations_delta.
    """
    if regen or not cache_manager.exists(specs):
        print(f"Generating data for:\n  PlotSpecs{astuple(specs)}")
        dataset = data_regen(specs, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return None
        cache_manager.commit(specs, dataset)
    return cache_manager.get(specs)


def data_regen(specs, cancel_event: Event):
    precision = min(specs.xmax - specs.xmin, specs.ymax - specs.ymin)
    use_dcomplex = np.isclose(precision, 0, atol=ATOL)

    C = dclingrid(specs) if use_dcomplex else clingrid(specs)
    closest_dataset = cache_manager.get_closest(specs)
    if closest_dataset is None:
        Z = dcomplex_zeroes(C.shape) if use_dcomplex else np.zeros_like(C, dtype=complex_type)
        iterations_delta = specs.iterations
    else:
        Z = closest_dataset.Z
        iterations_delta = specs.iterations - closest_dataset.to_specs().iterations

    C_chunks = np.array_split(C, PARALLELISM, axis=0)
    Z_chunks = np.array_split(Z, PARALLELISM, axis=0)
    with Pool(processes=CPU_CORES) as pool:
        results = pool.starmap_async(mandelbrot_calc_dcomplex if use_dcomplex else mandelbrot_calc, [(c, iterations_delta, z, cancel_event) for c, z in zip(C_chunks, Z_chunks)])
        # Merge back along rows
        diverging_order_chunks, mask_interior_chunks, Z_chunks = zip(*results.get())
    if cancel_event is not None and cancel_event.is_set():
        return None
    diverging_order = np.vstack(diverging_order_chunks)
    mask_interior = np.vstack(mask_interior_chunks)
    Z = np.vstack(Z_chunks)
    dataset = MandelbrotData(diverging_order, mask_interior, np.array(astuple(specs)), Z)
    return dataset


def dclingrid(specs: PlotSpecs):
    xmin, xmax, ymin, ymax = Decimal(specs.xmin), Decimal(specs.xmax), Decimal(specs.ymin), Decimal(specs.ymax)
    # Generate Decimal ranges
    x = [xmin + (xmax - xmin) * Decimal(i) / Decimal(specs.width - 1) for i in range(specs.width)]
    y = [ymin + (ymax - ymin) * Decimal(j) / Decimal(specs.height - 1) for j in range(specs.height)]

    # Create 2D array of DComplex
    C = np.empty((specs.height, specs.width), dtype=object)
    for j in range(specs.height):
        for i in range(specs.width):
            C[j, i] = DComplex(real=x[i], imag=y[j])
    return C
