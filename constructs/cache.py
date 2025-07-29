import glob
import os
from collections import defaultdict, OrderedDict
from dataclasses import astuple

import numpy as np

from constructs.model import PlotSpecs, MandelbrotData

FILE_PREFIX = 'tmp'


class CacheManager:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.dirname(__file__)
        self.directory = defaultdict(OrderedDict)

    def get_filename(self, specs: PlotSpecs):
        total_iterations = specs.iterations
        filename = f"{FILE_PREFIX}-{total_iterations}-{specs.xmin}-{specs.xmax}-{specs.ymin}-{specs.ymax}.npz"
        return os.path.join(self.cache_dir, filename)

    def cleanup(self):
        filename_pattern = f"{FILE_PREFIX}-*.npz"
        files = glob.glob(os.path.join(self.cache_dir, filename_pattern))
        for filename in files:
            os.remove(filename)

    def exists(self, specs: PlotSpecs) -> bool:
        key = specs.xmin, specs.xmax, specs.ymin, specs.ymax
        total_iterations = specs.iterations
        return key in self.directory and total_iterations in self.directory[key]

    def commit(self, specs: PlotSpecs, dataset: MandelbrotData):
        key = specs.xmin, specs.xmax, specs.ymin, specs.ymax
        total_iterations = specs.iterations
        filename = self.get_filename(specs)
        np.savez(filename, escapes=dataset.escapes, interior=dataset.interior, rect=np.array(astuple(specs)), Z=dataset.Z)
        self.directory[key][total_iterations] = filename

    def get(self, specs: PlotSpecs) -> MandelbrotData:
        filename = self.get_filename(specs)
        mandelbrot = np.load(filename)
        dataset = mandelbrot['escapes']
        interior = mandelbrot['interior']
        rect = np.array(mandelbrot['rect'])
        Z_payload = mandelbrot['Z']
        # Apply custom colormap for exterior
        return MandelbrotData(dataset, interior, rect, Z_payload)


cache_manager: CacheManager = CacheManager()


def cache_cleanup():
    cache_manager.cleanup()
