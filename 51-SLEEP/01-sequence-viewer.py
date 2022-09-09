from pictor import DaVinci
import matplotlib.pyplot as plt
import numpy as np
from mne.io import concatenate_raws, read_raw_edf
import mne
from scipy import signal
from pictor import Pictor



class SeqViewer(DaVinci):

  WIN_SIZE = 100000
  OVERLAP = 20000

  def __init__(self):
    super(SeqViewer, self).__init__('EGG signal')
    self.data = None
    self.data1 = None
    self.annot = None
    self.annot_des = None
    self.add_plotter(self.show_eeg_signal)
    # self.add_plotter(self.show_eeg_pz)
    self.signal_name1 = "C3A2"
    self.signal_name2 = "C4A1"

  # region: Public Methods

  # endregion: Public Methods

  def load_data(self):
    raw_data = read_raw_edf("data/ucddb/ucddb003.rec.edf", preload = True)
    raw_data_channel1 = raw_data.pick_channels([self.signal_name1])
    eeg_channel1 = raw_data_channel1.to_data_frame()  # 将读取的数据转换成pandas的DataFrame数据格式
    raw_data = read_raw_edf("data/ucddb/ucddb003.rec.edf", preload = True)
    raw_data_channel2 = raw_data.pick_channels([self.signal_name2])
    eeg_channel2 = raw_data_channel2.to_data_frame()  # 将读取的数据转换成pandas的DataFrame数据格式
    # raw_annot = mne.read_annotations("data/sc4001ec-hypnogram.edf")
    # print(raw_annot.to_data_frame())
    # annot = raw_annot.to_data_frame().to_dict()
    # self.annot = annot['duration']
    # self.annot_des = annot['description']
    eeg_channel1 = eeg_channel1.values[:, 1] # 转换成numpy的特有数据格式
    eeg_channel2 = eeg_channel2.values[:, 1] # 转换成numpy的特有数据格式
    return eeg_channel1, eeg_channel2
    #查看事件的id和事件


  def preprocess_data(self, data):
    #滤波
    b, a = signal.butter(7, 0.64)
    filted_data = signal.filtfilt(b, a, data)
    #归一化
    arr_mean = np.mean(filted_data)
    arr_std = np.std(filted_data)
    precessed_data = (filted_data - arr_mean) / arr_std
    return precessed_data


  def set_data(self, data, data1):
    assert isinstance(data, np.ndarray) and len(data.shape) == 1
    self.data = data
    self.data1 = data1
    print("************************************")
    print("data_length:", len(data))
    print("************************************")
    self.objects = np.arange(0, len(data) - self.WIN_SIZE, self.OVERLAP)


  def show_eeg_signal(self, x, ax: plt.Axes):
    indices = np.arange(x, x + self.WIN_SIZE)
    ax.plot(indices, self.data[indices], 'b')
    ax.plot(indices, self.data1[indices], 'r')
    annot = np.zeros(154)
    annot_des = [self.annot_des[i] for i in range(154)]
    sum = 0
    for i in range(154):
      sum += self.annot[i] * 100
      annot[i] = sum
    line_i = [i for i in indices if i in annot]
    for i in line_i: ax.plot([i, i], [-7, 7], 'r-')
    plt.ylim(ymin = -10, ymax = 10)
    for i, k in enumerate(line_i): ax.text(k + 200, 7, annot_des[i])


  # def show_eeg_signal_pro(self):
  #   p = Pictor.signal_viewer(default_win_size = 10000)
  #   p.objects = self.data
  #   print("***************************")
  #   print(type(self.data))
  #   print("***************************")
  #   p.show()

  def show_total(self):
    mapping = {"EEG Fpz-Cz":'eeg'}
    raw_train = read_raw_edf("../data/SC4001E0-PSG.edf", preload = True)
    annot_train = mne.read_annotations("./../data/SC4001EC-Hypnogram.edf")
    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)
    raw_train.plot(duration = 30, scalings = 'auto')
    plt.show()



if __name__ == '__main__':
  sv = SeqViewer()
  eeg_channel1, eeg_channel2 = sv.load_data()
  eeg_channel1_filted = sv.preprocess_data(eeg_channel1)
  eeg_channel2_filted = sv.preprocess_data(eeg_channel2)
  sv.set_data(eeg_channel1_filted, eeg_channel2_filted)
  sv.show()
  # sv.show_total()
  # sv.show_eeg_signal_pro()



