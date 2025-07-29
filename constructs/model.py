from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt, image, colorbar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, TextBox

PIXEL_X, PIXEL_Y = 2560, 1600


@dataclass(frozen=True)
class MandelbrotData:
    escapes: np.ndarray
    interior: np.ndarray
    rect: np.ndarray
    Z: np.ndarray = None

    def to_viz_data(self) -> 'MandelbrotViz':
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
        # Set interior (non-diverged) points to solid color (e.g., black)
        img[interior] = [0, 0, 0]  # dark interior
        return MandelbrotViz(img, CMAP_EXT, vmin, vmax, specs)


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


def iter_heuristic(rect):
    dx = rect.xmax - rect.xmin
    dy = rect.ymax - rect.ymin
    iterations = int(150 * np.log10(dx * dy) - 209 * np.log10(dx * dy) + 327)
    return min(iterations, 2048)


CMAP_EXT = LinearSegmentedColormap.from_list(
    "electric", ["#000428", "#004e92", "#00d4ff", "#ffffff"], N=1024
)
