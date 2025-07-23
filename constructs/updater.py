from constructs.data import data_gen, PlotSpecs, iter_heuristic
from constructs.history import HistoryHandler
from constructs.viz import PlotHandle, mandelbrot_viz


class DataRefreshHandler:
    def __init__(self, handle: PlotHandle, history_handle: HistoryHandler = None, regen=False):
        self.handle = handle
        self.history_handle = history_handle
        self.regen = regen

        self.timer = None
        self.delay = 200  # milliseconds
        self.latest_xlim = self.handle.ax.get_xlim()
        self.latest_ylim = self.handle.ax.get_ylim()
        self.xsid = self.handle.ax.callbacks.connect("xlim_changed", self._on_limit_change)
        self.ysid = self.handle.ax.callbacks.connect("ylim_changed", self._on_limit_change)
        self.handle.iter_box.on_submit(self._on_iteration_change)

    def _on_limit_change(self, event_ax):
        if self.latest_xlim == event_ax.get_xlim() and self.latest_ylim == event_ax.get_ylim():
            # skip identical limits
            return
        # Update the latest limits on every event
        self.latest_xlim = self.handle.ax.get_xlim()
        self.latest_ylim = self.handle.ax.get_ylim()
        rect = PlotSpecs(*self.latest_xlim + self.latest_ylim)
        self.handle.iterations = iter_heuristic(rect)
        self.handle.iter_box.set_val(str(self.handle.iterations))

        # Restart timer
        if self.timer is not None:
            self.timer.stop()
        self.timer = self.handle.ax.figure.canvas.new_timer(interval=self.delay)
        self.timer.add_callback(self._process_latest_limits)
        self.timer.start()

    def _process_latest_limits(self):
        print(f"Processing latest limits:\n  Rect{self.latest_xlim + self.latest_ylim} iterations: {self.handle.iterations}")
        specs = PlotSpecs(*self.latest_xlim, *self.latest_ylim, self.handle.iterations)
        new_data = data_gen(specs, regen=self.regen)
        handle = mandelbrot_viz(new_data, self.handle)
        # cbar is created new
        self.handle.cbar = handle.cbar
        self.timer = None  # reset timer

    def _on_iteration_change(self, text):
        try:
            iterations = int(text)
            if iterations == self.handle.iterations:
                return
        except ValueError:
            print(f'Invalid number: {text}')
            return

        old_iterations = self.handle.iterations
        if self.history_handle is not None:
            specs = PlotSpecs(*self.latest_xlim + self.latest_ylim, old_iterations)
            self.history_handle.append(specs)
        self.handle.iterations = iterations
        self._process_latest_limits()
