import glob
import os
from bisect import bisect_left, bisect
from collections import defaultdict, OrderedDict
from dataclasses import astuple

import numpy as np

from constructs.model import PlotSpecs, MandelbrotData

FILE_PREFIX = 'tmp'


class CacheManager:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.dirname(__file__)
        self.directory = defaultdict(dict)

    def gen_filename(self, specs: PlotSpecs):
        filename = f"{FILE_PREFIX}-{specs.iterations}-{specs.xmin}-{specs.xmax}-{specs.ymin}-{specs.ymax}.npz"
        return os.path.join(self.cache_dir, filename)

    def cleanup(self):
        filename_pattern = f"{FILE_PREFIX}-*.npz"
        files = glob.glob(os.path.join(self.cache_dir, filename_pattern))
        for filename in files:
            os.remove(filename)

    def exists(self, specs: PlotSpecs) -> bool:
        key = specs.xmin, specs.xmax, specs.ymin, specs.ymax
        return key in self.directory and specs.iterations in self.directory[key]

    def commit(self, specs: PlotSpecs, dataset: MandelbrotData):
        key = specs.xmin, specs.xmax, specs.ymin, specs.ymax
        filename = self.gen_filename(specs)
        np.savez(filename, escapes=dataset.escapes, interior=dataset.interior, specs=np.array(astuple(specs)), Z=dataset.Z)
        self.directory[key][specs.iterations] = filename

    def get(self, specs: PlotSpecs) -> MandelbrotData:
        filename = self.gen_filename(specs)
        return self.load(filename)

    def get_closest(self, specs: PlotSpecs) -> MandelbrotData:
        if self.exists(specs):
            return self.get(specs)
        key = specs.xmin, specs.xmax, specs.ymin, specs.ymax
        if key not in self.directory:
            return None
        entries = self.directory[key]
        iterations = sorted(entries.keys())
        index = bisect(iterations, specs.iterations) - 1
        if index < 0:
            return None
        filename = entries[iterations[index]]
        return self.load(filename)

    def load(self, filename):
        mandelbrot = np.load(filename)
        dataset = mandelbrot['escapes']
        interior = mandelbrot['interior']
        rect = np.array(mandelbrot['specs'])
        Z_payload = mandelbrot['Z']
        # Apply custom colormap for exterior
        return MandelbrotData(dataset, interior, rect, Z_payload)


cache_manager: CacheManager = CacheManager()


def cache_cleanup():
    cache_manager.cleanup()
