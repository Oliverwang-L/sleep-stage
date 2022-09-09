"""-> pictor.objects"""
from collections import OrderedDict
from roma.spqr.finder import walk
from mne.io import concatenate_raws, read_raw_edf
from scipy import signal
import numpy as np
import os



SIGNAL_ECG1 = "chan 1"
SIGNAL_ECG2 = "chan 2"
SIGNAL_ECG3 = "chan 3"
SIGNAL_ECG4 = "ECG"
SIGNAl_EEG1 = "C3A2"
SIGNAl_EEG2 = "C4A1"


class SleepData(object):

  def __init__(self, name: str):
    self.channels = OrderedDict()
    self.name = name


  # region: Builtin Data Reader

  @staticmethod
  def read_ucddb(path):
    """Under given `path` directory, files should be organized as in
       https://www.physionet.org/content/ucddb/1.0.0/

    :return: a list of SleepData objects
    """
    global eeg_channel1_filted, eeg_channel2_filted, ecg_channel4_filted
    # change *.rec to *.edf
    os.chdir(path)     # change dir to data directory
    edf_file_list = walk(path, 'file', '*lifecard.edf')

    # For each .edf file, create a SleepData object
    sleep_data_list = []
    for edf_file in edf_file_list:
      # Get file names
      prefix = edf_file[:-13]
      rec_file = prefix + '.rec'
      if os.path.exists(rec_file):
        new_name = prefix + '.edf'
        os.rename(rec_file, new_name)
      rec_file = prefix + '.edf'
      respevt_file = prefix + '_respevt.txt'
      stage_file = prefix + '_stage.txt'

      # create an object
      sd = SleepData(name=prefix.split('/')[-1])

      # Read edf file
      raw_edf = read_raw_edf(edf_file,preload = True)
      raw_edf_channel1 = raw_edf.pick_channels([SIGNAL_ECG1])
      ecg_channel1 = raw_edf_channel1.to_data_frame()  # 将读取的数据转换成pandas的DataFrame数据格式
      ecg_channel1 = ecg_channel1.values[:, 1] # 转换成numpy的特有数据格式
      ecg_channel1_filted = sd.preprocess_data(ecg_channel1)  # 数据预处理
      raw_edf = read_raw_edf(edf_file,preload = True)
      raw_edf_channel2 = raw_edf.pick_channels([SIGNAL_ECG2])
      ecg_channel2 = raw_edf_channel2.to_data_frame()  # 将读取的数据转换成pandas的DataFrame数据格式
      ecg_channel2 = ecg_channel2.values[:, 1] # 转换成numpy的特有数据格式
      ecg_channel2_filted = sd.preprocess_data(ecg_channel2)  # 数据预处理
      raw_edf = read_raw_edf(edf_file,preload = True)
      raw_edf_channel3 = raw_edf.pick_channels([SIGNAL_ECG3])
      ecg_channel3 = raw_edf_channel3.to_data_frame()  # 将读取的数据转换成pandas的DataFrame数据格式
      ecg_channel3 = ecg_channel3.values[:, 1] # 转换成numpy的特有数据格式
      ecg_channel3_filted = sd.preprocess_data(ecg_channel3)  # 数据预处理

      # Read rec file
      raw_rec = read_raw_edf(rec_file,preload=True)
      raw_rec_channel1 = raw_rec.pick_channels([SIGNAl_EEG1])
      eeg_channel1 = raw_rec_channel1.to_data_frame()
      eeg_channel1 = eeg_channel1.values[:, 1]
      eeg_channel1_filted = sd.preprocess_data(eeg_channel1)
      print("************************************")
      print(len(eeg_channel1_filted))
      print("************************************")
      raw_rec = read_raw_edf(rec_file,preload=True)
      raw_rec_channel2 = raw_rec.pick_channels([SIGNAl_EEG2])
      eeg_channel2 = raw_rec_channel2.to_data_frame()
      eeg_channel2 = eeg_channel2.values[:, 1]
      eeg_channel2_filted = sd.preprocess_data(eeg_channel2)
      raw_rec = read_raw_edf(rec_file,preload=True)
      raw_rec_channel3 = raw_rec.pick_channels([SIGNAL_ECG4])
      ecg_channel4 = raw_rec_channel3.to_data_frame()
      ecg_channel4 = ecg_channel4.values[:, 1]
      ecg_channel4_filted = sd.preprocess_data(ecg_channel4)


      # Read respevt file
      if os.path.exists(respevt_file):
        pass

      # Read stage file
      if os.path.exists(stage_file):
        pass

      sd.channels["ecg1"] = ecg_channel1_filted
      sd.channels["ecg2"] = ecg_channel2_filted
      sd.channels["ecg3"] = ecg_channel3_filted
      sd.channels["ecg4"] = ecg_channel4_filted
      sd.channels["eeg1"] = eeg_channel1_filted
      sd.channels["eeg2"] = eeg_channel2_filted
      sleep_data_list.append(sd)
      print("****************************", len(ecg_channel1_filted))
      break  # TODO: ol01
    return sleep_data_list


  def preprocess_data(self, data):
    #滤波
    b, a = signal.butter(7, 0.64)
    filted_data = signal.filtfilt(b, a, data)
    #归一化
    arr_mean = np.mean(filted_data)
    arr_std = np.std(filted_data)
    precessed_data = (filted_data - arr_mean) / arr_std
    return precessed_data
  # endregion: Builtin Data Reader



if __name__ == '__main__':
  import os
  abs_path = os.path.abspath(__file__)
  path = os.path.join(os.path.dirname(os.path.dirname(abs_path)),
                      'data', 'ucddb')
  _ = SleepData.read_ucddb(path)
