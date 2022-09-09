from collections import OrderedDict
from roma import console
from roma.spqr.finder import walk
from typing import List
from slp_set_loc import *
from misc import *

from tframe import DataSet
import numpy as np
import os
import pickle

from tframe.data.base_classes import DataAgent



class SleepRecord():
  """
  Data modalities:
  (1) EEG: EEG signals, usually has multiple channel, stored in .edf files;
           read as numpy arrays of shape [num_channels, seq_len]

  """

  FOLDER_NAMES = {
    'ucddb': 'ucddb',
    'sleepedf-train': 'sleep_edf\sleep-cassette-train',
    'sleepedf-test': 'sleep_edf\sleep-cassette-test',
    'sleepedf-correct': 'sleep_edf\sleep-cassette-correct',
    'apnea-train': 'apnea\osas-train',
    'apnea-test': 'apnea\osas-test',
  }

  PROPERTIES_SLEEP = {
    'CLASSES': ['Sleep stage W', 'Sleep stage R', 'Sleep stage 1',
                'Sleep stage 2', 'Sleep stage 3'],
    'NUM_CLASSES': 5
  }

  PROPERTIES_APNEA = {
    'CLASSES': ['A', 'N'],
    'NUM_CLASSES': 2
  }


  def __init__(self):
    self.data_dict = OrderedDict()

  class Keys:
    EEG = 'EEG'
    EOG = 'EOG'
    STAGES = 'STAGES'
    APNEAECG = 'APNEAECG'

  # region: Public Mehtods

  def report_detail(self):
    console.show_info(f'{self.record_id}:')
    for k, v in self.data_dict.items():
      console.supplement(f'{k}.shape = {v.shape}')

  # endregion: Public Mehtods

  # region: Data IO
  @classmethod
  def load(cls, path):
    from slp_core import th
    data_path = os.path.join(path, 'sleep_edf\sleep-cassette-train')
    tfds_fn = os.path.join(path, 'data.tfds')
    if os.path.exists(tfds_fn) and os.stat(tfds_fn).st_size != 0:
      with open(tfds_fn, 'rb') as file:
        console.show_status('Loading `{}` ...'.format(tfds_fn))
        return pickle.load(file)
    sleep_data_list = cls.read_sleep_records(data_path)
    preprocess_data_list = DataProcess.preprocess_data(sleep_data_list)
    # print(len(preprocess_data_list[0].data_dict['eeg'][:, 1]))
    data_set = cls.convert_to_dataset(preprocess_data_list)
    data_set.save(tfds_fn)

  @classmethod
  def convert_to_dataset(cls, preprocess_data_list):
    from slp_core import th
    dataset_len = len(preprocess_data_list)
    print('*' * 30)
    print('Patient number: ', dataset_len)
    feature = []
    targets = []
    PROPERTIES = cls.PROPERTIES_SLEEP
    for sr_index in range(0, dataset_len):
      data_person, ann_person = DataProcess.sleep_data_reshape(
        preprocess_data_list[sr_index])
      feature.append(data_person)
      targets.append(ann_person)
    data_set = SleepSet(feature, targets, **PROPERTIES)
    return data_set

  @classmethod
  def read_sleep_records(cls, data_dir: str, data_config=None):
    # sanity check
    assert os.path.exists(data_dir)

    num_records = None
    # if .recs file exists, load it directly
    folder_name = os.path.split(data_dir)[-1]
    recs_fn = os.path.join(data_dir, folder_name + '-xai{}.recs'.format(
      f'-{data_config}' if data_config else ''))

    if os.path.exists(recs_fn) and os.stat(recs_fn).st_size != 0:
      # with open support closing file in any situation
      with open(recs_fn, 'rb') as file:
        console.show_status('Loading `{}` ...'.format(recs_fn))
        return pickle.load(file)

    # Otherwise, read sleep records according to folder name
    if folder_name in ['ucddb']:
      sleep_records = cls._read_ucddb(data_dir)
    elif folder_name in ['sleep-cassette-test', 'sleep-cassette-train']:
      if data_config: num_records = int(data_config)
      sleep_records = cls._read_sleep_edf(data_dir, num_records)
    elif folder_name in ['osas-train', 'osas-test']:
      if data_config: num_records = int(data_config)
      sleep_records = cls._read_osas(data_dir, num_records)
    else:
      raise KeyError(f'!! Unknown dataset `{folder_name}`')

    # Save file and return
    with open(recs_fn, 'wb') as file:
      pickle.dump(sleep_records, file, pickle.HIGHEST_PROTOCOL)
    return sleep_records

  # region: Builtin IO Methods

  #region: respiratroty task
  @classmethod
  def _read_osas(cls, data_dir: str, num_records: int):
    import random
    # Define static variables
    console.show_status('Reading `apnea-ecg` dataset ...')

    # judge database is test or train
    # Create an empty list
    sleep_records: List[cls] = []

    # Get all .dat files
    dat_file_list: List[str] = walk(data_dir, 'file', '???.dat')
    if num_records is not None:
      dat_file_list = dat_file_list[:num_records]

    # Read records one by one
    for dat_fn in dat_file_list:
      # Get id
      id = os.path.split(dat_fn)[-1].split('.')[0]
      if id == 'x01':
        # shuffle the object
        random.shuffle(sleep_records)
      ecg_path = os.path.join(data_dir, id)
      sr = SleepRecord()

      console.show_status(f'Reading record `{id}` ...')
      console.split()

      # (1) Read dat
      ecg_data= cls._read_dat_file(ecg_path)
      # (2) Read annotation
      ecg_anno, ecg_anno_sample = cls._read_apn_file(ecg_path)

      sr.data_dict['APNEAECG'] = ecg_data
      sr.data_dict['ANNOTATION'] = ecg_anno
      sr.data_dict['SAMPLE'] = ecg_anno_sample
      # Append this record to list
      sleep_records.append(sr)

    return sleep_records

  @classmethod
  def _read_dat_file(cls, fn: str):
    import wfdb
    ecg_data = wfdb.rdrecord(fn)
    return ecg_data.p_signal
    pass

  @classmethod
  def _read_apn_file(cls, fn: str):
    import wfdb
    anno = wfdb.rdann(fn, 'apn')
    return np.array(anno.symbol), anno.sample

  #endregion


  #region: sleep stage task
  @classmethod
  def _read_edf_data_file(cls, fn: str, channel_list: List[str], *args) -> np.ndarray:
    from mne.io import concatenate_raws, read_raw_edf
    from mne.io.edf.edf import RawEDF
    from slp_core import th
    with read_raw_edf(fn, preload=True) as raw_edf:
      assert isinstance(raw_edf, RawEDF)
      # edf_data[:, 0] is time, edf_data[:, k] is channel k (k >= 1)
      edf_data_raw = raw_edf.pick_channels(channel_list).to_data_frame().values
      if args is not None:
        edf_data = edf_data_raw[args[0] * th.random_sample_length :args[1] * th.random_sample_length]
      else:
        edf_data = edf_data_raw
      assert isinstance(edf_data, np.ndarray)
    return edf_data

  @classmethod
  def _read_edf_anno_file(cls, fn: str):
    from mne import read_annotations

    print(fn)
    stage_anno_raw = []
    start_index = 0
    end_index = 0
    # with read_annotations(fn) as raw_anno:
    #   anno = raw_anno.to_data_frame().values
    #   anno_dura = anno[:, 1]
    #   anno_desc = anno[:, 2]
    raw_anno = read_annotations(fn)
    anno = raw_anno.to_data_frame().values
    anno_dura = anno[:, 1]
    anno_desc = anno[:, 2]
    for dura_num in range(len(anno_dura) - 1):
      for stage_num in range(int(anno_dura[dura_num]) // 30):
        stage_anno_raw.append(anno_desc[dura_num])
    raw_length = len(stage_anno_raw)
    for stage_index, stage_desc in enumerate(stage_anno_raw):
      if stage_desc == 'Sleep stage W':
        continue
      else:
        start_index = stage_index - 60
        break
    for stage_index, stage_desc in enumerate(stage_anno_raw[::-1]):
      if stage_desc == 'Sleep stage W':
        continue
      else:
        end_index = raw_length - stage_index + 60
        break
    stage_anno = stage_anno_raw[start_index:end_index]
    return np.array(stage_anno), start_index, end_index

  @classmethod
  def _read_ucddb(cls, data_dir: str):
    # define static variables
    class Channels:
      EEG = {'SIGNAl_EEG1': "C3A2",
             'SIGNAl_EEG2': "C4A1" }
      ECG = {'SIGNAL_ECG1': "chan 1",
             'SIGNAL_ECG2': "chan 2",
             'SIGNAL_ECG3': "chan 3",
             'SIGNAL_ECG4': "ECG"}

    console.show_status('reading `ucddb` dataset ...')

    # create an empty list
    sleep_records: list[cls] = []

    # get all .edf files
    rec_file_list: list[str] = walk(data_dir, 'file', '*.rec*')

    # Read records one by one
    for rec_fn in rec_file_list:
      # Get id
      id = os.path.split(rec_fn)[-1].split('.r')[0]
      sr = SleepRecord()

      console.show_status(f'Reading record `{id}` ...')
      console.split()

      # (1) Read EEG data
      # Rename .rec file if necessary
      if rec_fn[-3:] != 'edf':
        os.rename(rec_fn, rec_fn + '.edf')
        rec_fn = rec_fn + '.edf'

      sr.data_dict['EEG'] = cls._read_edf_data_file(
        rec_fn, list(Channels.EEG.values()))
      console.split()

      # (2) Read ECG data
      fn = os.path.join(data_dir, id + '_lifecard.edf')
      sr.data_dict['ECG'] = cls._read_edf_data_file(
        fn, list(Channels.ECG.values()))
      console.split()

      # (3) Read stage labels
      fn = os.path.join(data_dir, id + '_stage.txt')
      assert os.path.exists(fn)
      with open(fn, 'r') as stage:
        stage_ann = [line.strip() for line in stage.readlines()]
      sr.data_dict['stage'] = np.array(stage_ann)

      # (4) Read misc data
      fn = os.path.join(data_dir, id + '_respevt.txt')
      assert os.path.exists(fn)
      with open(fn, 'r') as resp:
        resp_content = resp.readlines()
        resp_time = [line.strip().split()[0] for line in resp_content if ':' in line]
        resp_anno = [line.strip().split()[1] for line in resp_content if ':' in line]
      sr.data_dict['resp'] = np.array([resp_time,resp_anno])
      # Append this record to list
      sleep_records.append(sr)

    # Return records
    return sleep_records

  @classmethod
  def _read_sleep_edf(cls, data_dir: str, num_records: int):
    # Define static variables
    class Channels:
      EEG = {'SIGNAl_EEG1': "EEG Fpz-Cz",
             'SIGNAl_EEG2': "EEG Pz-Oz" }
      EOG = {'SIGNAL EOG': "EOG horizontal"}

    console.show_status('Reading `sleep_edf` dataset ...')

    # Create an empty list
    sleep_records: List[cls] = []

    # Get all .edf files
    hypnogram_file_list: List[str] = walk(data_dir, 'file', '*Hypnogram*')
    if num_records is not None:
      hypnogram_file_list = hypnogram_file_list[:num_records]

    # Read records one by one
    for Hypnogram_fn in hypnogram_file_list:
      # Get id
      id = os.path.split(Hypnogram_fn)[-1].split('-')[0]
      sr = SleepRecord()

      console.show_status(f'Reading record `{id}` ...')
      console.split()

      # (1) Read label
      # Rename label file if necessary
      # if Hypnogram_fn[7] != '0':
      #   os.rename(Hypnogram_fn, Hypnogram_fn[:7] + '0' + '-Hypnogram.edf')
      #   id = Hypnogram_fn[:7] + '0'
      #   Hypnogram_fn = id + '-Hypnogram.edf'

      # sr.data_dict['stage'], begin, end = cls._read_edf_anno_file(Hypnogram_fn)
      annotation, begin, end = cls._read_edf_anno_file(Hypnogram_fn)

      # (2) Read PSG
      fn = os.path.join(data_dir, id[:7] + '0' + '-PSG.edf')
      assert os.path.exists(fn)
      # sr.data_dict['EEG'] = cls._read_edf_data_file(fn, list(Channels.EEG.values()), begin, end)
      eeg_data = cls._read_edf_data_file(fn, list(Channels.EEG.values()), begin, end)

      eog_data = cls._read_edf_data_file(fn, list(Channels.EOG.values()), begin, end)

      sr.data_dict['stage'] = annotation
      sr.data_dict['EEG'] = eeg_data
      sr.data_dict['EOG'] = eog_data
      console.split()

      stage_num = sr.data_dict['stage'].size
      eeg_num = sr.data_dict['EEG'].shape[0]
      eog_num = sr.data_dict['EOG'].shape[0]

      assert stage_num == eeg_num / 3000 == eog_num / 3000

      # Append this record to list
      sleep_records.append(sr)

    # Return records
    return sleep_records

  #endregion

  @classmethod
  def _split_and_return(cls, data_set, train_size, validate_size, test_size,
                        over_classes=False):
    from tframe.data.dataset import DataSet
    # assert isinstance(data_set, DataSet)
    names, sizes = [], []
    for name, size in zip(['Train Set', 'Validation Set', 'Test Set'],
                          [train_size, validate_size, test_size]):
      if size == 0: continue
      names.append(name)
      sizes.append(size)
    data_sets = data_set.split(*sizes, over_classes=over_classes, names=names)
    # Show data info
    # cls._show_data_sets_info(data_sets)
    return data_sets
  # endregion: Builtin IO Methods

  # endregion: Data IO


if __name__ == '__main__':
  # Get data path
  abs_path = os.path.abspath(__file__)
  data_dir = os.path.join(os.path.dirname(os.path.dirname(abs_path)),
                          'data', 'sleep_edf', 'sleep-cassette')

  #
  # sleep_records = SleepRecord.read_sleep_records(data_dir)
  # for slp_rec in sleep_records: slp_rec.report_detail()
  SleepRecord.load(data_dir)

