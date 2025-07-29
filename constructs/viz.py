from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox

from constructs.model import PlotHandle, MandelbrotData


def static_buttons(fig):
    # Add a TextBox widget for iteration input
    axbox = plt.axes((0.36, 0.92, 0.08, 0.03))  # [left, bottom, width, height]
    iter_box = TextBox(axbox, 'Enter number:', initial='0')

    # Add Undo button
    ax_undo = fig.add_axes((0.45, 0.92, 0.08, 0.03))  # [left, bottom, width, height]
    btn_undo = Button(ax_undo, 'Undo')

    # Add Reset button
    ax_reset = fig.add_axes((0.54, 0.92, 0.08, 0.03))  # [left, bottom, width, height]
    btn_reset = Button(ax_reset, 'Reset')

    # Add Redo button
    ax_redo = fig.add_axes((0.63, 0.92, 0.08, 0.03))
    btn_redo = Button(ax_redo, 'Redo')
    return btn_undo, btn_reset, btn_redo, iter_box


def mandelbrot_viz(mandelData: MandelbrotData = None, handle: PlotHandle = None) -> PlotHandle:
    assert mandelData is not None or handle is not None
    if mandelData is None:
        mandelData = handle.data
    viz_data = mandelData.to_viz_data()
    if handle is None:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        im = ax.imshow(
            viz_data.img,
            origin='lower',
            cmap=viz_data.cmap,
            vmin=viz_data.vmin, vmax=viz_data.vmax,
            extent=(viz_data.specs.xmin, viz_data.specs.xmax, viz_data.specs.ymin, viz_data.specs.ymax),
        )
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("Escape Time (Smoothed Iterations)")
        btn_undo, btn_reset, btn_redo, iter_box = static_buttons(fig)
        handle = PlotHandle(fig, ax, im, cbar, btn_undo, btn_reset, btn_redo, iter_box, iterations=viz_data.specs.iterations, data=mandelData)
        handle.update_iter_box(viz_data.specs.iterations)
        return handle

    fig, ax, im = handle.fig, handle.ax, handle.im
    im.set_data(viz_data.img)
    im.set_extent(ax.get_xlim() + ax.get_ylim())
    fig.canvas.draw_idle()

    # Remove the old colorbar
    handle.cbar.remove()
    # Add colorbar
    handle.cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    handle.cbar.set_label("Escape Time (Smoothed Iterations)")
    # plt.axis("off")
    # plt.savefig(f"{filename}.png", dpi=600, bbox_inches="tight", pad_inches=0)
    handle.data = mandelData
    handle.iterations = viz_data.specs.iterations
    return handle
