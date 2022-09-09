"""-f pictor.plotters"""
from pictor.plotters import Plotter

import matplotlib.pyplot as plt
import numpy as np
from pictor_sleep.sleep_data import SleepData



class Monitor(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Monitor, self).__init__(self.show_data, pictor)

  # x = [SleepData(), SleepData(), ]
  def show_data(self, x: SleepData, fig: plt.Figure):
    ax1: plt.Axes = fig.add_subplot(311)
    ax1.plot(x.channels["eeg1"])
    ax1.set_title("EEG_C3A2")
    ax2: plt.Axes = fig.add_subplot(312)
    ax2.plot(x.channels["eeg2"])
    ax2.set_title("EEG_C4A1")
    ax3: plt.Axes = fig.add_subplot(313)
    ax3.plot(x.channels["ecg4"])
    ax3.set_title("ECG")

  def register_shortcuts(self):
    # self.register_a_shortcut('h', lambda: print('hahhahahah'),
    #                          description='Laugh')
    pass

