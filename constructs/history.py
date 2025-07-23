from constructs.data import PlotSpecs, data_gen
from constructs.viz import PlotHandle, mandelbrot_viz


class HistoryHandler:
    def __init__(self, handle: PlotHandle, regen=False):
        self.history = [PlotSpecs(*handle.ax.get_xlim(), *handle.ax.get_ylim(), iterations=handle.iterations)]
        self.index = 0
        self.handle = handle
        self.regen = regen
        self.init_specs = self.history[0]

        self.handle.btn_undo.on_clicked(self.undo)
        self.handle.btn_reset.on_clicked(self.reset)
        self.handle.btn_redo.on_clicked(self.redo)

    def reset(self, event):
        print(f"Reset event")
        specs = self.init_specs
        self.history = [specs]
        self.index = 0

        self.handle.ax.set_xlim(specs.xmin, specs.xmax)
        self.handle.ax.set_ylim(specs.ymin, specs.ymax)
        self.handle.iter_box.set_val(str(specs.iterations))
        self.handle.fig.canvas.draw_idle()

        new_data = data_gen(specs, regen=self.regen)
        handle = mandelbrot_viz(new_data, self.handle)
        # cbar is created new
        self.handle.cbar = handle.cbar

    def undo(self, event):
        print(f"Undo event at index {self.index} (history length: {len(self.history)})")
        if self.index > 0:
            self.index -= 1
            specs = self.history[self.index]
            # Set new limits centered on click
            self.handle.ax.set_xlim(specs.xmin, specs.xmax)
            self.handle.ax.set_ylim(specs.ymin, specs.ymax)
            self.handle.iter_box.set_val(str(specs.iterations))
            self.handle.fig.canvas.draw_idle()

            new_data = data_gen(specs, regen=self.regen)
            handle = mandelbrot_viz(new_data, self.handle)
            # cbar is created new
            self.handle.cbar = handle.cbar
        else:
            print("Undo event ignored for history is empty.")

    def redo(self, event):
        print(f"Redo event at index {self.index} (history length: {len(self.history)})")
        if self.index < len(self.history) - 1:
            self.index += 1
            specs = self.history[self.index]
            self.handle.ax.set_xlim(specs.xmin, specs.xmax)
            self.handle.ax.set_ylim(specs.ymin, specs.ymax)
            self.handle.iter_box.set_val(str(specs.iterations))
            self.handle.fig.canvas.draw_idle()

            new_data = data_gen(specs, regen=self.regen)
            handle = mandelbrot_viz(new_data, self.handle)
            # cbar is created new
            self.handle.cbar = handle.cbar
        else:
            print("Redo click event ignored for future is empty.")

    def append(self, specs: PlotSpecs):
        while self.index < len(self.history) - 1:
            self.history.pop()
        self.history.append(specs)
        self.index += 1
