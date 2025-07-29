from constructs.calc import data_gen
from constructs.viz import mandelbrot_viz
from constructs.model import PlotHandle, PlotSpecs


class HistoryCtrl:
    def __init__(self, handle: PlotHandle, regen=False):
        self.history = [PlotSpecs(*handle.ax.get_xlim(), *handle.ax.get_ylim(), iterations=handle.iterations)]
        self.index = 0
        self.handle = handle
        self.regen = regen
        self.init_specs = self.history[0]

    def reset(self, event):
        print(f"Reset event")
        specs = self.init_specs
        self.history = [specs]
        self.index = 0

        self.handle.ax.set_xlim(specs.xmin, specs.xmax)
        self.handle.ax.set_ylim(specs.ymin, specs.ymax)
        self.handle.update_iter_box(specs.iterations)
        self.handle.fig.canvas.draw_idle()

        new_data = data_gen(specs, regen=self.regen)
        handle = mandelbrot_viz(new_data, self.handle)
        # cbar is created new
        self.handle.cbar = handle.cbar

    def undo(self, event):
        if self.index > 0:
            self.index -= 1
            print(f"Undo event: use specs {self.index + 1}/{len(self.history)}")
            specs = self.history[self.index]
            # Set new limits centered on click
            self.handle.ax.set_xlim(specs.xmin, specs.xmax)
            self.handle.ax.set_ylim(specs.ymin, specs.ymax)
            self.handle.update_iter_box(specs.iterations)
            self.handle.fig.canvas.draw_idle()

            new_data = data_gen(specs, regen=self.regen)
            handle = mandelbrot_viz(new_data, self.handle)
            # cbar is created new
            self.handle.cbar = handle.cbar
        else:
            print("Undo event ignored-history is empty.")

    def redo(self, event):
        if self.index < len(self.history) - 1:
            self.index += 1
            print(f"Redo event: use specs {self.index + 1}/{len(self.history)}")
            specs = self.history[self.index]
            self.handle.ax.set_xlim(specs.xmin, specs.xmax)
            self.handle.ax.set_ylim(specs.ymin, specs.ymax)
            self.handle.update_iter_box(specs.iterations)
            self.handle.fig.canvas.draw_idle()

            new_data = data_gen(specs, regen=self.regen)
            handle = mandelbrot_viz(new_data, self.handle)
            # cbar is created new
            self.handle.cbar = handle.cbar
        else:
            print("Redo event ignored-already latest.")

    def append(self, specs: PlotSpecs):
        while self.index < len(self.history) - 1:
            self.history.pop()
        self.history.append(specs)
        self.index += 1
