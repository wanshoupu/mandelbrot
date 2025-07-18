import matplotlib.pyplot as plt

from constructs.history import HistoryHandler
from constructs.updater import DataRefreshHandler
from constructs.data import data_gen, Rect, cache_cleanup
from constructs.viz import mandelbrot_viz, PlotHandle
from constructs.zoomer import ZoomHandler


def interactive_plot():
    rect = Rect(-2.0, 1.0, -1.5, 1.5)
    # rect = Rect(np.float64(-1.9449859417539945), np.float64(-1.944985936166059), np.float64(-2.94131147638115e-09), np.float64(2.646623971311721e-09))
    # rect = Rect(np.float64(-1.9449859385), np.float64(-1.9449859375), np.float64(-.5e-09), np.float64(.5e-09))
    # rect = Rect(np.float64(-1.9449859379728804), np.float64(-1.9449859379572556), np.float64(-5.681818181817935e-12), np.float64(9.943181818182066e-12))
    # rect = Rect(np.float64(-1.9449859379344914), np.float64(-1.944985937926679), np.float64(5.225243506493812e-12), np.float64(1.3037743506493811e-11))
    # rect = Rect(np.float64(-1.9449859379303545), np.float64(-1.9449859379301093), np.float64(1.2128871240657437e-11), np.float64(1.2373011865657429e-11))

    data = data_gen(rect, regen=False)
    plot = mandelbrot_viz(data)
    history_handle = HistoryHandler(plot)
    zoom_handler = ZoomHandler(plot, zoom_factor=0.8, history_handle=history_handle)
    handler = DataRefreshHandler(plot, regen=False)
    plt.show()


def static_plot():
    rects = [
        Rect(-2.0, 1.0, -1.5, 1.5),
        Rect(-.7, -.5, -.5, -.3),
        Rect(-0.7, -0.6, -.4, -.3),
        Rect(-0.68, -0.64, -.38, -.34),
    ]
    for rect in rects:
        data = data_gen(rect)
        handle = mandelbrot_viz(data)
        plt.show()


if __name__ == "__main__":
    # static_plot()
    interactive_plot()
    cache_cleanup()
