import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')  # Set the backend to Qt5

import matplotlib.pyplot as plt
from collections import deque


class GraphGUI:

    def __init__(self):
        plt.ion()

        self.start = 0
        self.visible = 662

        # initialize deques
        self.dy1 = deque(np.zeros(self.visible), self.visible)
        self.dy2 = deque(np.zeros(self.visible), self.visible)
        self.dy3 = deque(np.zeros(self.visible), self.visible)
        self.dy4 = deque(np.zeros(self.visible), self.visible)
        self.dy5 = deque(np.zeros(self.visible), self.visible)
        self.store1 = []
        self.store2 = []
        self.store3 = []
        self.dx = deque(np.zeros(self.visible), self.visible)

        # get interval of entire time frame
        self.interval = np.linspace(-self.visible, 100000, num=100000)

        # define figure size
        self.fig = plt.figure(figsize=(12, 10))

        # define axis1, labels, and legend
        self.ah1 = self.fig.add_subplot(411)
        self.ah1.set_ylabel("Delta X", fontsize=14)
        self.l1, = self.ah1.plot(self.dx, self.dy1, color='rosybrown', label="abs(Delta X)")
        self.ah1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

        # define axis2, labels, and legend
        self.ah2 = self.fig.add_subplot(412)
        self.ah2.set_ylabel("Delta Y", fontsize=14)
        self.l2, = self.ah2.plot(self.dx, self.dy2, color='green', label="abs(Delta Y)")
        self.ah2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

        # define axis3, labels, and legend
        self.ah3 = self.fig.add_subplot(413)
        self.ah3.set_xlabel("Time step", fontsize=14, labelpad=10)
        self.ah3.set_ylabel("Delta Theta", fontsize=14)
        self.l3, = self.ah3.plot(self.dx, self.dy3, color='blue', label="abs(Delta Theta)")
        self.ah3.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

        # define axis3, labels, and legend
        self.ah4 = self.fig.add_subplot(414)
        self.ah4.set_xlabel("Time step", fontsize=14, labelpad=10)
        self.ah4.set_ylabel("Velocity", fontsize=14)
        self.l4, = self.ah4.plot(self.dx, self.dy4, color='cyan', label="Velocity Left", alpha=0.5)
        self.l5, = self.ah4.plot(self.dx, self.dy5, color='purple', label="Velocity Right", alpha=0.5)
        self.ah4.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

    def update_plot(self, data):
        # extend deques (both x and y axes)
        self.dy1.extend(data["delta_x"])
        self.dy2.extend(data["delta_y"])
        self.dy3.extend(np.degrees(data["delta_theta"]))
        self.dy4.extend(data["vl"])
        self.dy5.extend(data["vr"])
        self.store1.extend(data["delta_x"])
        self.store2.extend(data["delta_y"])
        self.store3.extend(np.degrees(data["delta_theta"]))
        self.dx.extend(self.interval[self.start:self.start + self.visible])

        # update plot
        self.l1.set_ydata(self.dy1)
        self.l2.set_ydata(self.dy2)
        self.l3.set_ydata(self.dy3)
        self.l4.set_ydata(self.dy4)
        self.l5.set_ydata(self.dy5)
        self.l1.set_xdata(self.dx)
        self.l2.set_xdata(self.dx)
        self.l3.set_xdata(self.dx)
        self.l4.set_xdata(self.dx)
        self.l5.set_xdata(self.dx)

        # get mean of deques
        mdy1 = np.mean(self.dy1)
        mdy2 = np.mean(self.dy2)
        mdy3 = np.mean(self.dy3)

        # set x- and y-limits based on their mean
        self.ah1.set_ylim(-5, 30)
        self.ah1.set_xlim(self.interval[self.start], self.interval[self.start + self.visible])
        self.ah2.set_ylim(-5, 30)
        self.ah2.set_xlim(self.interval[self.start], self.interval[self.start + self.visible])
        self.ah3.set_ylim(-5, 20)
        self.ah3.set_xlim(self.interval[self.start], self.interval[self.start + self.visible])
        self.ah4.set_ylim(-6, 6)
        self.ah4.set_xlim(self.interval[self.start], self.interval[self.start + self.visible])

        # update start
        self.start += 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
