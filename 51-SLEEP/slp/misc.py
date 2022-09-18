from slp_agent_loc import *
from slp_set_loc import *
import numpy as np

class DataProcess(object):

  def __init__(self):
    self.eeg_data = []
    self.ecg_data = []


  # region: preprocess data
  @classmethod
  def preprocess_data(cls, sleep_data_list):
    for sleep_data in sleep_data_list:
      if 'EEG' in sleep_data.data_dict.keys():
        eeg_data = sleep_data.data_dict['EEG']
        sleep_data.data_dict['EEG'][:, 1] = cls.preprocess_sleep_data(eeg_data[:, 1])
        if eeg_data.shape[1] == 3:
          sleep_data.data_dict['EEG'][:, 2] = cls.preprocess_sleep_data(eeg_data[:, 2])
      if 'ECG' in sleep_data.data_dict.keys():
        ecg_data = sleep_data.data_dict['ECG']
        sleep_data.data_dict['ECG'][:, 1] = cls.preprocess_sleep_data(ecg_data[:, 1])
        sleep_data.data_dict['ECG'][:, 2] = cls.preprocess_sleep_data(ecg_data[:, 2])
        sleep_data.data_dict['ECG'][:, 3] = cls.preprocess_sleep_data(ecg_data[:, 3])
      if 'EOG' in sleep_data.data_dict.keys():
        eog_data = sleep_data.data_dict['EOG']
        sleep_data.data_dict['EOG'][:, 1] = cls.preprocess_sleep_data(eog_data[:, 1])
      if 'APNEAECG' in sleep_data.data_dict.keys():
        sleep_data.data_dict['APNEAECG'], sleep_data.data_dict['ANNOTATION'] = \
                                           cls.preprocess_apnea_data(sleep_data)
    # print("*************************************")
    # print("stage_ann:", sleep_data_list[0].data_dict['stage'][:10])
    # print("*************************************")
    return sleep_data_list

  @classmethod
  def preprocess_apnea_data(cls, data):
    from scipy import signal
    ecg_data = data.data_dict['APNEAECG'][:, 0]
    ecg_ann = data.data_dict['ANNOTATION']
    #filter
    b, a= signal.butter(2, [0.01, 0.7], 'bandpass')  # 0.5 - 35 Hz
    filted_data = signal.filtfilt(b, a, ecg_data)

    #Align data and labels
    end_index = data.data_dict['SAMPLE'][-1] + 6000
    if end_index > filted_data.shape[0]:
      end_index = data.data_dict['SAMPLE'][-2] + 6000
      ecg_ann = ecg_ann[:-1]
    aligned_data = filted_data[:end_index]
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print('end_index:', end_index)
    # print('label num:', data.data_dict['ANNOTATION'].shape[0])
    # print('aligned_data length:', aligned_data.shape[0])
    assert aligned_data.shape[0] // 6000 == ecg_ann.shape[0]
    return aligned_data, ecg_ann

  @classmethod
  def preprocess_sleep_data(cls, data):
    import numpy as np
    from scipy import signal
    #滤波
    b, a= signal.butter(7, 0.7, 'lowpass')
    filted_data = signal.filtfilt(b, a, data)
    #归一化
    arr_mean = np.mean(filted_data)
    arr_std = np.std(filted_data)
    precessed_data = (filted_data - arr_mean) / arr_std
    return precessed_data
  #endregion

  @classmethod
  def apnea_data_reshape(cls, data_person):
    label_dict = {'A':[1,0], 'N':[0,1]}
    apnea_ann_one_hot = []
    apnea_ecg = data_person.data_dict['APNEAECG']
    apnea_ann = data_person.data_dict['ANNOTATION']
    for label in apnea_ann:
      if label == 'N':
        apnea_ann_one_hot.extend(label_dict[label])
      elif label == 'A':
        apnea_ann_one_hot.extend(label_dict[label])
    apnea_ecg_reshape = apnea_ecg.reshape(apnea_ecg.shape[0] // 6000, 6000, 1)
    apnea_ann_reshape = np.array(apnea_ann_one_hot).reshape(len(apnea_ann_one_hot)
                                                            // 2, 2)
    return apnea_ecg_reshape, apnea_ann_reshape

  @classmethod
  def apnea_get_rr(cls, data_person, ann_person, sr_index = 0):
    from biosppy.signals import ecg
    from hrvanalysis import remove_outliers
    from hrvanalysis import remove_ectopic_beats
    from hrvanalysis import interpolate_nan_values
    from scipy import interpolate

    rr_intervals = []
    ann = []
    for num in range(data_person.shape[0]):
      # get R-R intervals from one minute raw data: ms
      r_peaks_index = ecg.hamilton_segmenter(data_person[num][:, 0], 100)
      r_peaks_index = r_peaks_index[0]
      r_peak = data_person[num][:, 0][r_peaks_index]
      rr = list(np.diff(r_peaks_index) * 10)
      # filter abnormal point
      # wipe_outliers = remove_outliers(rr_intervals=rr, low_rri=600, high_rri=1200)
      # rr = interpolate_nan_values(rr_intervals=wipe_outliers, interpolation_method='linear')
      # wipe_abnormal = remove_ectopic_beats(rr_intervals=rr, method='malik')
      # rr = interpolate_nan_values(rr_intervals=wipe_abnormal, interpolation_method='linear')
      # cubic interpolation at 3Hz
      ir = 2 # INTERPOLATION RATE(2HZ)
      time_range= 60 # 60-s INTERVALS OF ECG SIGNALS
      f_length = int(ir * time_range)
      tm_rr = np.linspace(0, time_range, num=len(rr))
      tm_rp = np.linspace(0, time_range, num=len(r_peak))
      tm_new = np.arange(0, (time_range), step=(1) / float(ir))
      if len(rr) < 40 or len(rr) >= 160:
        # r_peaks_index = ecg.hamilton_segmenter(data_person[num][:, 0], 100)
        # r_peaks_index = r_peaks_index[0]
        # rr = (r_peaks_index[1:] - r_peaks_index[:-1]) * 10
        # rr = list(rr)
        # if len(rr) <= 50 or len(rr) >= 160:
        print('!' * 40)
        print(len(data_person[num][:, 0]))
        print(r_peaks_index)
        print(sr_index, num)
        print('!' * 40)
        continue
      # print('>>>> rr length: ', len(rr))
      if len(rr) != len(tm_rr):
        print(len(rr), len(tm_rr))
      frr = interpolate.interp1d(tm_rr, rr, 'cubic')
      rr_interpolation = frr(tm_new)[:, np.newaxis]
      frp = interpolate.interp1d(tm_rp, r_peak, 'cubic')
      rpeak_interpolation = frp(tm_new)[:, np.newaxis]
      osa_data = np.hstack((rr_interpolation, rpeak_interpolation))
      rr_intervals.extend(osa_data[:])
      ann.extend(list(ann_person[num]))
    rr_intervals = np.array(rr_intervals).reshape(len(rr_intervals) // f_length,
                                                  f_length, 2)
    ann = np.array(ann).reshape(len(ann) // 2, 2)
    print('>>>> transform No.{} person to rr_intervals'.format(sr_index + 1))
    return rr_intervals, ann

  @classmethod
  def sleep_data_reshape(cls, data_person):
    label_dict = {'Sleep stage W':[1,0,0,0,0],
                  'Sleep stage R':[0,1,0,0,0],
                  'Sleep stage 1':[0,0,1,0,0],
                  'Sleep stage 2':[0,0,0,1,0],
                  'Sleep stage 3':[0,0,0,0,1],
                  'Sleep stage 4':[0,0,0,0,1],}
    sleep_stage_one_hot = []
    sleep_data_aasm = []
    sleep_eeg = data_person.data_dict['EEG'][:, 1][:, np.newaxis]
    sleep_eog = data_person.data_dict['EOG'][:, 1][:, np.newaxis]
    sleep_data = np.hstack((sleep_eeg, sleep_eog))
    sleep_stage = data_person.data_dict['stage']
    for index, stage_label in enumerate(sleep_stage):
      if stage_label in ['Movement time', 'Sleep stage ?']:
        continue
      sleep_data_aasm.extend(sleep_data[index * 3000:(index+1) * 3000])
      sleep_stage_one_hot.extend(label_dict[stage_label])
      if stage_label == 'Sleep stage 1':
        for i in range(2):
          sleep_stage_one_hot.extend(label_dict[stage_label])
          offset = np.random.randint(-200, 200)
          sleep_data_aasm.extend(sleep_data[index * 3000 + offset:(index+1) * 3000
                                + offset])
      # if stage_label == 'Sleep stage R':
      #   sleep_stage_one_hot.extend(label_dict[stage_label])
      #   offset = np.random.randint(-200, 200)
      #   sleep_data_aasm.extend(sleep_eeg[index * 3000 + offset:(index+1) * 3000
      #                                                         + offset])
    sleep_data_aasm = np.array(sleep_data_aasm)
    sleep_data_reshape = sleep_data_aasm.reshape(sleep_data_aasm.shape[0] // 3000, 3000, 2)
    sleep_stage_reshape = np.array(sleep_stage_one_hot).reshape(len(sleep_stage_one_hot) // 5, 5)

    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # print(sleep_data_reshape.shape)
    # print(sleep_stage_reshape.shape)
    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    return sleep_data_reshape, sleep_stage_reshape



if __name__ == '__main__':
  pass

