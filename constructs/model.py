from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt, image, colorbar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, TextBox

MAX_ITERATIONS = 2048
PIXEL_X, PIXEL_Y = 2560, 1600
CMAP_EXT = LinearSegmentedColormap.from_list(
    "electric", ["#000428", "#004e92", "#00d4ff", "#ffffff"], N=1024
)


@dataclass
class PlotSpecs:
    xmin: np.longfloat
    xmax: np.longfloat
    ymin: np.longfloat
    ymax: np.longfloat
    iterations: int = None
    width: int = PIXEL_X
    height: int = PIXEL_Y

    def __post_init__(self):
        if self.iterations is None:
            self.iterations = iter_heuristic(self)
        else:
            self.iterations = int(self.iterations)
        self.width, self.height = int(self.width), int(self.height)


@dataclass(frozen=True)
class MandelbrotData:
    escapes: np.ndarray
    interior: np.ndarray
    specs: np.ndarray
    Z: np.ndarray = None

    def to_viz_data(self) -> 'MandelbrotViz':
        escapes = self.escapes
        interior = self.interior
        specs = PlotSpecs(*self.specs)
        pixelx, pixely = escapes.shape

        # Create image array
        img = np.zeros((pixelx, pixely, 3))
        # Normalize exterior values for coloring
        vmin, vmax = escapes.min(), escapes.max()
        norm_div = (escapes / vmax) if vmax > 0 else escapes
        # Apply colormap to diverged (exterior) points
        img[~interior] = CMAP_EXT(norm_div[~interior])[:, :3]  # drop alpha
        # Set interior (non-diverged) points to solid color (e.g., black)
        img[interior] = [0, 0, 0]  # dark interior
        return MandelbrotViz(img, CMAP_EXT, vmin, vmax, specs)

    def to_specs(self) -> PlotSpecs:
        return PlotSpecs(*self.specs)


@dataclass
class PlotHandle:
    fig: plt.Figure
    ax: plt.Axes
    im: image.AxesImage
    cbar: colorbar.Colorbar
    btn_undo: Button
    btn_reset: Button
    btn_redo: Button
    iter_box: TextBox
    iterations: int
    data: MandelbrotData = None

    def update_iter_box(self, iterations, events_on=False):
        self.iterations = iterations
        if self.iter_box is None:
            return
        box: TextBox = self.iter_box
        eventson = box.eventson
        box.eventson = events_on
        box.set_val(str(iterations))
        box.eventson = eventson


@dataclass(frozen=True)
class MandelbrotViz:
    img: np.ndarray
    cmap: LinearSegmentedColormap
    vmin: float
    vmax: float
    specs: PlotSpecs


def iter_heuristic(rect: PlotSpecs):
    dx = rect.xmax - rect.xmin
    dy = rect.ymax - rect.ymin
    try:
        iterations = int(150 * np.log10(dx * dy) - 209 * np.log10(dx * dy) + 327)
    except:
        iterations = MAX_ITERATIONS
    return min(iterations, MAX_ITERATIONS)
