from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt, image, colorbar
from matplotlib.colors import LinearSegmentedColormap

from constructs.data import Rect, MandelbrotData
from matplotlib.widgets import Button


@dataclass
class PlotHandle:
    fig: plt.Figure
    ax: plt.Axes
    im: image.AxesImage
    cbar: colorbar.Colorbar
    btn_undo: Button
    btn_reset: Button
    btn_redo: Button


@dataclass
class MandelbrotViz:
    img: np.ndarray
    cmap: LinearSegmentedColormap
    vmin: float
    vmax: float
    rect: Rect


def to_viz_data(mandelData: MandelbrotData) -> MandelbrotViz:
    dataset = mandelData.dataset
    interior = mandelData.interior
    rect = Rect(*mandelData.rect)
    pixelx, pixely = dataset.shape
    cmap_ext = LinearSegmentedColormap.from_list(
        "electric", ["#000428", "#004e92", "#00d4ff", "#ffffff"], N=1024
    )

    # Create image array
    img = np.zeros((pixelx, pixely, 3))
    # Normalize exterior values for coloring
    vmin, vmax = dataset.min(), dataset.max()
    norm_div = dataset / vmax
    # Apply colormap to diverged (exterior) points
    img[~interior] = cmap_ext(norm_div[~interior])[:, :3]  # drop alpha
    # Set interior (non-diverged) points to solid color (e.g., black or red)
    img[interior] = [0, 0, 0]  # deep red interior
    return MandelbrotViz(img, cmap_ext, vmin, vmax, rect)


def static_buttons(fig):
    # Add Undo button
    ax_undo = fig.add_axes((0.4, 0.92, 0.1, 0.05))  # [left, bottom, width, height]
    btn_undo = Button(ax_undo, 'Undo')

    # Add Reset button
    ax_reset = fig.add_axes((0.52, 0.92, 0.1, 0.05))  # [left, bottom, width, height]
    btn_reset = Button(ax_reset, 'Reset')

    # Add Redo button
    ax_redo = fig.add_axes((0.64, 0.92, 0.1, 0.05))
    btn_redo = Button(ax_redo, 'Redo')
    return btn_undo, btn_reset, btn_redo


def mandelbrot_viz(mandelData: MandelbrotData, handle: PlotHandle = None) -> PlotHandle:
    viz_data = to_viz_data(mandelData)
    if handle is None:
        fig, ax = plt.subplots()
        im = ax.imshow(
            viz_data.img,
            origin='lower',
            cmap=viz_data.cmap,
            vmin=viz_data.vmin, vmax=viz_data.vmax,
            extent=(viz_data.rect.xmin, viz_data.rect.xmax, viz_data.rect.ymin, viz_data.rect.ymax),
        )
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("Escape Time (Smoothed Iterations)")
        btn_undo, btn_reset, btn_redo = static_buttons(fig)
        return PlotHandle(fig, ax, im, cbar, btn_undo, btn_reset, btn_redo)

    fig, ax, im = handle.fig, handle.ax, handle.im
    im.set_data(viz_data.img)
    im.set_extent(ax.get_xlim() + ax.get_ylim())
    fig.canvas.draw_idle()

    # Define the key press event handler
    def on_key(event):
        if event.key == 'f':  # press 'f' to toggle fullscreen
            fig.canvas.manager.full_screen_toggle()

    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Remove the old colorbar
    handle.cbar.remove()
    # Add colorbar
    handle.cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    handle.cbar.set_label("Escape Time (Smoothed Iterations)")
    # plt.axis("off")
    # plt.savefig(f"{filename}.png", dpi=600, bbox_inches="tight", pad_inches=0)
    return handle
