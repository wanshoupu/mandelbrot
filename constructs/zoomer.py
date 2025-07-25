import threading

from constructs.data import PlotSpecs, data_gen
from constructs.history import HistoryHandler
from constructs.viz import PlotHandle, mandelbrot_viz

DEBOUNCE_TIME = .1


class ZoomHandler:
    def __init__(self, plot_handle: PlotHandle, zoom_factor=0.5, history_handle: HistoryHandler = None, regen=False):
        self.handle = plot_handle
        self.zoom_factor = zoom_factor
        self.cid = self.handle.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.sid = self.handle.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.handle.iter_box.on_submit(self._on_iteration_change)

        self.history_handle = history_handle
        self.timer = None
        self.delay = 200  # milliseconds
        self.scroll_lock = threading.Lock()
        self.scroll_accumulator = 0
        self.regen = regen

    def on_click(self, event):
        if event.inaxes != self.handle.ax or event.button != 1:
            return  # Only respond to left-clicks inside the plot

        x, y = event.xdata, event.ydata
        print(f"Recentering at: ({x:.3f}, {y:.3f})")

        # Current window size
        x0, x1 = self.handle.ax.get_xlim()
        y0, y1 = self.handle.ax.get_ylim()
        dx = (x1 - x0) / 2
        dy = (y1 - y0) / 2

        # Set new limits centered on click
        specs = PlotSpecs(x - dx, x + dx, y - dy, y + dy, self.handle.iterations)
        self.handle.ax.set_xlim(specs.xmin, specs.xmax)
        self.handle.ax.set_ylim(specs.ymin, specs.ymax)
        self.handle.fig.canvas.draw_idle()

        new_data = data_gen(specs, regen=self.regen)
        handle = mandelbrot_viz(new_data, self.handle)
        # cbar is created new
        self.handle.cbar = handle.cbar

        if self.history_handle is not None:
            self.history_handle.append(specs)

    def on_scroll(self, event):
        if event.inaxes != self.handle.ax:
            return

        with self.scroll_lock:
            self.scroll_accumulator += event.step  # step is +1/-1 per scroll
            last_event = event  # save latest event
            if self.timer is not None:
                self.timer.cancel()

            self.timer = threading.Timer(
                DEBOUNCE_TIME, lambda: self.flush_scroll(last_event)
            )
            self.timer.start()

    def flush_scroll(self, event):
        with self.scroll_lock:
            steps = self.scroll_accumulator
            self.scroll_accumulator = 0
        if not steps:
            return

        scale_factor = self.zoom_factor ** steps

        ax = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        specs = PlotSpecs(xdata - new_width / 2, xdata + new_width / 2, ydata - new_height / 2, ydata + new_height / 2, self.handle.iterations)
        ax.set_xlim(specs.xmin, specs.xmax)
        ax.set_ylim(specs.ymin, specs.ymax)
        event.canvas.draw_idle()

        new_data = data_gen(specs, regen=self.regen)
        handle = mandelbrot_viz(new_data, self.handle)
        # cbar is created new
        self.handle.cbar = handle.cbar

        if self.history_handle is not None:
            self.history_handle.append(specs)

    def _on_iteration_change(self, text):
        try:
            iterations = int(text)
            if iterations == self.handle.iterations:
                return
            iteration_delta = iterations - self.handle.iterations
        except ValueError:
            print(f'Invalid number: {text}')
            return
        if iteration_delta < 0:
            specs = PlotSpecs(*self.handle.ax.get_xlim() + self.handle.ax.get_ylim(), iterations)
            self.handle.data = data_gen(specs, regen=self.regen)
        else:
            specs = PlotSpecs(*self.handle.ax.get_xlim() + self.handle.ax.get_ylim(), self.handle.iterations)
            Z_init = self.handle.data.Z
            self.handle.data = data_gen(specs, regen=self.regen, iterations_delta=iteration_delta, Z=Z_init)
            self.handle.iterations += iteration_delta

        handle = mandelbrot_viz(handle=self.handle)

        # cbar is created new
        self.handle.cbar = handle.cbar

        if self.history_handle is not None:
            self.history_handle.append(specs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import TextBox, Button

    fig, ax = plt.subplots()
    data = np.random.rand(100, 100)
    im = ax.imshow(data, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    axbox = plt.axes((0.36, 0.92, 0.08, 0.03))  # [left, bottom, width, height]
    iter_box = TextBox(axbox, 'Enter number:', initial='0')
    handle = PlotHandle(fig, ax, im, cbar, Button(axbox, 'Undo'), Button(axbox, 'Undo'), Button(axbox, 'Undo'), iter_box=iter_box, iterations=10, data=data)
    zh = ZoomHandler(handle, .5)
    plt.title("Scroll anywhere in plot")
    plt.show()
