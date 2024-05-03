import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')  # Set the backend to Qt5

import matplotlib.pyplot as plt
from collections import deque


class GraphGUI:

    def __init__(self):
        plt.ion()

        self.start = 0
        self.visible = 500

        # initialize deques
        self.dy1 = deque(np.zeros(self.visible), self.visible)
        self.dy2 = deque(np.zeros(self.visible), self.visible)
        self.dx = deque(np.zeros(self.visible), self.visible)

        # get interval of entire time frame
        self.interval = np.linspace(-self.visible, 100000, num=100000)

        # define figure size
        self.fig = plt.figure(figsize=(10, 6))

        # define axis1, labels, and legend
        self.ah1 = self.fig.add_subplot(211)
        self.ah1.set_ylabel("Tracking variable 1", fontsize=14)
        self.l1, = self.ah1.plot(self.dx, self.dy1, color='rosybrown', label="Variable 1")
        self.ah1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

        # define axis2, labels, and legend
        self.ah2 = self.fig.add_subplot(212)
        self.ah2.set_xlabel("Time step", fontsize=14, labelpad=10)
        self.ah2.set_ylabel("Tracking variable 1", fontsize=14)
        self.l2, = self.ah2.plot(self.dx, self.dy2, color='silver', label="Variable 2")
        self.ah2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

    def update_plot(self, data):
        # extend deques (both x and y axes)
        self.dy1.append(data["vl"])
        self.dy2.append(data["vr"])
        self.dx.extend(self.interval[self.start:self.start + self.visible])

        # update plot
        self.l1.set_ydata(self.dy1)
        self.l2.set_ydata(self.dy2)
        self.l1.set_xdata(self.dx)
        self.l2.set_xdata(self.dx)

        # get mean of deques
        mdy1 = np.mean(self.dy1)
        mdy2 = np.mean(self.dy2)

        # set x- and y-limits based on their mean
        self.ah1.set_ylim(-5, 5)
        self.ah1.set_xlim(self.interval[self.start], self.interval[self.start + self.visible])
        self.ah2.set_ylim(-5, 5)
        self.ah2.set_xlim(self.interval[self.start], self.interval[self.start + self.visible])

        # update start
        self.start += 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
