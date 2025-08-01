import threading
from multiprocessing import Event

from constructs.calc import data_gen
from constructs.history import HistoryCtrl
from constructs.model import PlotHandle, PlotSpecs
from constructs.viz import mandelbrot_viz

DEBOUNCE_TIME = .1
MIN_ZOOM_LEVEL = 1e-14


class MandelbrotCtrl:
    def __init__(self, plot_handle: PlotHandle, zoom_factor=0.5, history_handle: HistoryCtrl = None, regen=False, cancel_event: Event = None):
        self.handle = plot_handle
        self.zoom_factor = zoom_factor
        self.cid = self.handle.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.sid = self.handle.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.history_handle = history_handle
        if self.history_handle is not None:
            self.handle.btn_undo.on_clicked(self.history_handle.undo)
            self.handle.btn_reset.on_clicked(self.history_handle.reset)
            self.handle.btn_redo.on_clicked(self.history_handle.redo)

        # Connect the key press event
        self.kid = self.handle.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.handle.iter_box.on_submit(self._on_iteration_change)

        self.timer = None
        self.delay = 200  # milliseconds
        self.scroll_lock = threading.Lock()
        self.scroll_accumulator = 0
        self.regen = regen
        self.cancel_event: Event = cancel_event

    # Define the key press event handler
    def on_key(self, event):
        # print(event.key, event.ctrl, event.shift, event.alt)
        if event.key == 'f':  # press 'f' to toggle fullscreen
            self.handle.fig.canvas.manager.full_screen_toggle()
        elif event.key == 'cmd+z':
            self.history_handle.undo(None)
        elif event.key == 'cmd+Z':
            self.history_handle.redo(None)
        elif event.key == 'cmd+r':
            self.history_handle.reset(None)
        elif event.key == 'cmd+c' or event.key == 'ctrl+c':
            if self.cancel_event is not None:
                self.cancel_event.set()

    def on_click(self, event):
        if event.inaxes != self.handle.ax or event.button != 1:
            return  # Only respond to left-clicks inside the plot
        if self.cancel_event is not None:
            self.cancel_event.clear()
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

        new_data = data_gen(specs, regen=self.regen, cancel_event=self.cancel_event)
        if new_data is not None:
            mandelbrot_viz(new_data, self.handle)
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

        if self.cancel_event is not None:
            self.cancel_event.clear()

        ax = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        scale_factor = self.zoom_factor ** steps
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor
        if min(new_width, new_height) < MIN_ZOOM_LEVEL:
            print(f'Zoom level is already the lowest {MIN_ZOOM_LEVEL:.3f}')
            return

        specs = PlotSpecs(xdata - new_width / 2, xdata + new_width / 2, ydata - new_height / 2, ydata + new_height / 2)
        self.handle.update_iter_box(specs.iterations)
        ax.set_xlim(specs.xmin, specs.xmax)
        ax.set_ylim(specs.ymin, specs.ymax)
        event.canvas.draw_idle()

        new_data = data_gen(specs, regen=self.regen, cancel_event=self.cancel_event)
        if new_data is not None:
            mandelbrot_viz(new_data, self.handle)
            if self.history_handle is not None:
                self.history_handle.append(specs)

    def _on_iteration_change(self, text):
        try:
            iterations = int(text)
            if iterations == self.handle.iterations:
                return
        except ValueError:
            print(f'Invalid number: {text}')
            return

        if self.cancel_event is not None:
            self.cancel_event.clear()

        specs = PlotSpecs(*self.handle.ax.get_xlim() + self.handle.ax.get_ylim(), iterations)
        new_data = data_gen(specs, regen=self.regen, cancel_event=self.cancel_event)
        if new_data is not None:
            mandelbrot_viz(new_data, handle=self.handle)
            if self.history_handle is not None:
                self.history_handle.append(specs)
