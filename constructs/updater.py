import numpy as np

from constructs.data import data_gen, Rect
from constructs.viz import PlotHandle, mandelbrot_viz


class DataRefreshHandler:
    def __init__(self, handle: PlotHandle, regen=False):
        self.handle = handle
        self.regen = regen

        self.timer = None
        self.delay = 200  # milliseconds
        self.latest_xlim = self.handle.ax.get_xlim()
        self.latest_ylim = self.handle.ax.get_ylim()
        self.xsid = self.handle.ax.callbacks.connect("xlim_changed", self._on_limit_change)
        self.ysid = self.handle.ax.callbacks.connect("ylim_changed", self._on_limit_change)
        self.handle.iter_box.on_submit(self._on_iteration_change)

    def _on_limit_change(self, event_ax):
        # Update the latest limits on every event
        self.latest_xlim = self.handle.ax.get_xlim()
        self.latest_ylim = self.handle.ax.get_ylim()
        rect = Rect(*self.latest_xlim + self.latest_ylim)
        self.handle.iterations = self.iter_heuristic(rect)
        self.handle.iter_box.set_val(str(self.handle.iterations))

        # Restart timer
        if self.timer is not None:
            self.timer.stop()
        self.timer = self.handle.ax.figure.canvas.new_timer(interval=self.delay)
        self.timer.add_callback(self._process_latest_limits)
        self.timer.start()

    def _process_latest_limits(self):
        print(f"Processing latest limits:\n  Rect({self.latest_xlim + self.latest_ylim}) iterations: {self.handle.iterations}")
        rect = Rect(*self.latest_xlim + self.latest_ylim)
        new_data = data_gen(rect, self.handle.iterations, regen=self.regen)
        handle = mandelbrot_viz(new_data, self.handle)
        # cbar is created new
        self.handle.cbar = handle.cbar
        self.timer = None  # reset timer

    def _on_iteration_change(self, text):
        try:
            self.handle.iterations = int(text)
        except ValueError:
            print(f'Invalid number: {text}')

        self._process_latest_limits()

    def iter_heuristic(self, rect):
        dx = rect.xmax - rect.xmin
        dy = rect.ymax - rect.ymin
        ex = int(-np.log(dx * dy))
        iterations = 1 << max(7, ex)
        return iterations
