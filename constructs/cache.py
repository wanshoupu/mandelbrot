import glob
import os

FILE_PREFIX = 'tmp'


class CacheManager:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.dirname(__file__)

    def get_filename(self, specs):
        filename = f"{FILE_PREFIX}-{specs.iterations}-{specs.xmin}-{specs.xmax}-{specs.ymin}-{specs.ymax}.npz"
        return os.path.join(self.cache_dir, filename)

    def cleanup(self):
        filename_pattern = f"{FILE_PREFIX}-*.npz"
        files = glob.glob(os.path.join(self.cache_dir, filename_pattern))
        for filename in files:
            os.remove(filename)


cache_manager = CacheManager()
