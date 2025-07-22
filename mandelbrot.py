import matplotlib.pyplot as plt

from constructs.history import HistoryHandler
from constructs.updater import DataRefreshHandler
from constructs.data import data_gen, Rect, cache_cleanup
from constructs.viz import mandelbrot_viz, PlotHandle
from constructs.zoomer import ZoomHandler


def interactive_plot():
    rect = Rect(-2.8, 2.0, -1.5, 1.5)
    # rect = Rect((-1.9449859417539945), (-1.944985936166059), (-2.94131147638115e-09), (2.646623971311721e-09))
    # rect = Rect((-1.9449859385), (-1.9449859375), (-.5e-09), (.5e-09))
    # rect = Rect((-1.9449859379728804), (-1.9449859379572556), (-5.681818181817935e-12), (9.943181818182066e-12))
    # rect = Rect((-1.9449859379344914), (-1.944985937926679), (5.225243506493812e-12), (1.3037743506493811e-11))
    # rect = Rect((-1.9449859379303545), (-1.9449859379301093), 1.2128871240657437e-11, 1.2373011865657429e-11)
    # rect = Rect(-1.9449855034094694, -1.9449855034080101, 6.126053995240807e-08, 6.126199514393089e-08)
    # rect = Rect(-1.9449855034748507, -1.94498550337357, 6.12237434943789e-08, 6.132471769024717e-08)
    # rect = Rect(-1.9449855073419413, -1.9449854963610607, 5.511244475221234e-08, 6.60600890047455e-08)
    # rect = Rect(-0.38798823799911153, -0.35965403910189353, -0.6667188599577806, -0.6490099856470192)

    data = data_gen(rect, regen=False)
    plot = mandelbrot_viz(data)
    history_handle = HistoryHandler(plot)
    zoom_handler = ZoomHandler(plot, zoom_factor=0.8, history_handle=history_handle)
    handler = DataRefreshHandler(plot, regen=False)
    plt.tight_layout(pad=0.1)
    plt.show()


def static_plot():
    rects = [
        Rect(-2.0, 1.0, -1.5, 1.5),
        Rect(-.7, -.5, -.5, -.3),
        Rect(-0.7, -0.6, -.4, -.3),
        Rect(-0.68, -0.64, -.38, -.34),
    ]
    for rect in rects:
        data = data_gen(rect, 100, regen=False)
        handle = mandelbrot_viz(data)
        plt.show()


if __name__ == "__main__":
    # static_plot()
    interactive_plot()
    cache_cleanup()
