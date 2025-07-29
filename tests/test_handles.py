import matplotlib
import numpy as np
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import TextBox, Button

from constructs.model import PlotHandle, MandelbrotData
from constructs.controller import MandelbrotCtrl

matplotlib.use('Qt5Agg')  # or 'QtAgg'


def test_zoom_left_click(qtbot):
    # Prepare data and figure
    data = np.random.rand(50, 50)
    extent = (-2, 2, -2, 2)

    fig, ax = plt.subplots()
    im = ax.imshow(data, extent=extent, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    axbox = plt.axes((0.36, 0.92, 0.08, 0.03))  # [left, bottom, width, height]
    iter_box = TextBox(axbox, 'Enter number:', initial='0')
    handle = PlotHandle(fig, ax, im, cbar, Button(axbox, 'Undo'), Button(axbox, 'Undo'), Button(axbox, 'Undo'), iter_box=iter_box, iterations=10, data=MandelbrotData(data, data, extent))
    zoom_handler = MandelbrotCtrl(handle, zoom_factor=0.8)

    # Wrap canvas in QWidget for qtbot
    canvas = FigureCanvas(fig)
    qtbot.addWidget(canvas)

    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()

    # Simulate a left click in the center of the plot (0, 0 in data coords)
    qtbot.mouseClick(canvas, Qt.LeftButton, pos=canvas.mapFromGlobal(canvas.mapToGlobal(canvas.rect().center())))

    # Optionally, assert something after the zoom
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # the plot is re-centered on the click point
    actual_xlim = sum(xlim)
    actual_ylim = sum(ylim)
    assert np.isclose(actual_xlim, 0.8185483870967731), f'actual_xlim: {actual_xlim}'
    assert np.isclose(actual_ylim, 0.07359307359307365), f'actual_ylim: {actual_ylim}'

    # the limit is not changed
    assert np.isclose(xlim[1] - xlim[0], xlim_before[1] - xlim_before[0])
    assert np.isclose(ylim[1] - ylim[0], ylim_before[1] - ylim_before[0])

    canvas.close()
