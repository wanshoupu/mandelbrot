from fractals.mandelbrot.data import Rect
from fractals.mandelbrot.viz import PlotHandle


class HistoryHandler:
    def __init__(self, handle: PlotHandle):
        self.history = [Rect(*handle.ax.get_xlim() + handle.ax.get_ylim())]
        self.index = 0
        self.handle = handle

        self.handle.btn_undo.on_clicked(self.undo)
        self.handle.btn_reset.on_clicked(self.reset)
        self.handle.btn_redo.on_clicked(self.redo)

    def reset(self, event):
        print(f"Reset event")
        rect = Rect(-2.0, 1.0, -1.5, 1.5)
        self.history = [rect]
        self.index = 0

        self.handle.ax.set_xlim(rect.xmin, rect.xmax)
        self.handle.ax.set_ylim(rect.ymin, rect.ymax)
        self.handle.fig.canvas.draw_idle()

    def undo(self, event):
        print(f"Undo event")
        if self.index > 0:
            self.index -= 1
            rect = self.history[self.index]
            # Set new limits centered on click
            self.handle.ax.set_xlim(rect.xmin, rect.xmax)
            self.handle.ax.set_ylim(rect.ymin, rect.ymax)
            self.handle.fig.canvas.draw_idle()
        else:
            print("Undo event ignored for history is empty.")

    def redo(self, event):
        print(f"Redo event")
        if self.index < len(self.history) - 1:
            self.index += 1
            rect = self.history[self.index]
            self.handle.ax.set_xlim(rect.xmin, rect.xmax)
            self.handle.ax.set_ylim(rect.ymin, rect.ymax)
            self.handle.fig.canvas.draw_idle()
        else:
            print("Redo click event ignored for future is empty.")

    def append(self, rect: Rect):
        self.history.append(rect)
        self.index += 1