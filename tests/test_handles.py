import matplotlib
import numpy as np
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from constructs.model import PlotHandle
from constructs.zoomer import ZoomHandler

matplotlib.use('Qt5Agg')  # or 'QtAgg'


def test_zoom_left_click(qtbot):
    # Prepare data and figure
    data = np.random.rand(50, 50)
    extent = (-2, 2, -2, 2)

    fig, ax = plt.subplots()
    im = ax.imshow(data, extent=extent, origin='lower', cmap='inferno')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    handle = PlotHandle(fig, ax, im, cbar, None, None, None)
    zoom_handler = ZoomHandler(handle, zoom_factor=0.8)

    # Wrap canvas in QWidget for qtbot
    canvas = FigureCanvas(fig)
    qtbot.addWidget(canvas)

    canvas.show()

    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()

    # Simulate a left click in the center of the plot (0, 0 in data coords)
    qtbot.mouseClick(canvas, Qt.LeftButton, pos=canvas.mapFromGlobal(canvas.mapToGlobal(canvas.rect().center())))

    # Optionally, assert something after the zoom
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # the plot is re-centered on the click point
    assert np.isclose(sum(xlim), 0.7010228166797798)
    assert np.isclose(sum(ylim), 0.07359307359307365)

    # the limit is not changed
    assert np.isclose(xlim[1] - xlim[0], xlim_before[1] - xlim_before[0])
    assert np.isclose(ylim[1] - ylim[0], ylim_before[1] - ylim_before[0])

    canvas.close()


def test_zoom_right_click(qtbot):
    # Prepare data and figure
    data = np.random.rand(50, 50)
    extent = (-2, 2, -2, 2)

    fig, ax = plt.subplots()
    im = ax.imshow(data, extent=extent, origin='lower', cmap='inferno')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    handle = PlotHandle(fig, ax, im, cbar, None, None, None)
    zoom_handler = ZoomHandler(handle, zoom_factor=0.5)

    # Wrap canvas in QWidget for qtbot
    canvas = FigureCanvas(fig)
    qtbot.addWidget(canvas)

    canvas.show()

    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()

    # Simulate a left click in the center of the plot (0, 0 in data coords)
    qtbot.mouseClick(canvas, Qt.RightButton, pos=canvas.mapFromGlobal(canvas.mapToGlobal(canvas.rect().center())))

    # Optionally, assert something after the zoom
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # nothing changes
    assert xlim_before == xlim
    assert ylim_before == ylim

    canvas.close()



