import matplotlib.pyplot as plt

from constructs.data import data_gen, cache_cleanup
from constructs.model import PlotSpecs, iter_heuristic
from constructs.history import HistoryHandler
from constructs.viz import mandelbrot_viz
from constructs.zoomer import ZoomHandler


def interactive_plot():
    specs = PlotSpecs(-2.8, 2.0, -1.5, 1.5)
    # specs = PlotSpecs(-0.7513642549206841, -0.7513415106902602, -0.02865126476442289, -0.028637049620407924)

    data = data_gen(specs, regen=False)
    plot = mandelbrot_viz(data)
    history_handle = HistoryHandler(plot)
    zoom_handler = ZoomHandler(plot, zoom_factor=.6, history_handle=history_handle, regen=False)
    plt.tight_layout(pad=0.1)
    plt.show()


def fit_iter(rects):
    for rect in sorted(rects, key=lambda rect: rect.iterations):
        cit = iter_heuristic(rect)
        print(cit, rect.iterations)


def static_plot(rects):
    for rect in rects:
        data = data_gen(rect, regen=True)
        plot = mandelbrot_viz(data)
        history_handle = HistoryHandler(plot)
        zoom_handler = ZoomHandler(plot, history_handle=history_handle)
        plt.show()


if __name__ == "__main__":
    # rects = [
    #     PlotSpecs(0.35939168473296146, 0.35939168484073236, -0.6147586102408348, -0.614758610173478, 5000),
    #     PlotSpecs(-1.9449859417539945, -1.944985936166059, -2.94131147638115e-09, 2.646623971311721e-09, 500),
    #     PlotSpecs(-1.9449859385, -1.9449859375, -5e-10, 5e-10, 200),
    #     PlotSpecs(-1.9449859379344914, -1.944985937926679, 5.225243506493812e-12, 1.3037743506493811e-11, 200),
    #     PlotSpecs(-1.9449855034094694, -1.9449855034080101, 6.126053995240807e-08, 6.126199514393089e-08, 1000),
    #     PlotSpecs(-1.9449855034748507, -1.94498550337357, 6.12237434943789e-08, 6.132471769024717e-08, 2000),
    #     PlotSpecs(-1.9449855073419413, -1.9449854963610607, 5.511244475221234e-08, 6.60600890047455e-08, 1500),
    #     PlotSpecs(-0.38798823799911153, -0.35965403910189353, -0.6667188599577806, -0.6490099856470192, 100),
    #     PlotSpecs(0.2507056737353899, 0.25074074977468586, 3.4443255829038355e-05, 5.636578038900195e-05, 1000),
    #     PlotSpecs(0.4368069999763344, 0.4398493614168821, -0.35852693900808164, -0.35662546310773924, 500),
    #     PlotSpecs(-0.7461263814442011, -0.7453139017908547, -0.11031698787565258, -0.10980918809231052, 1886),
    # ]
    #
    # fit_iter(rects)
    # static_plot(rects)
    try:
        interactive_plot()
    finally:
        cache_cleanup()
