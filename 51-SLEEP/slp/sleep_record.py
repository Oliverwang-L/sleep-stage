# from collections import OrderedDict
# from roma import console
# from roma.spqr.finder import walk
# from typing import List
# from slp_set import *
# from misc import *
#
# from tframe import DataSet
# import numpy as np
# import os
# import pickle
#
# from tframe.data.base_classes import DataAgent
#
# class SleepRecord():
#   """
#   Data modalities:
#   (1) EEG: EEG signals, usually has multiple channel, stored in .edf files;
#            read as numpy arrays of shape [num_channels, seq_len]
#
#   """
#
#   DATA_NAME = 'Sleep-edf-cassette'
#   PROPERTIES = {
#     'CLASSES': ['Sleep stage W', 'Sleep stage R', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage M'],
#     'NUM_CLASSES': 7
#   }
#
#
#   def __init__(self):
#     self.data_dict = OrderedDict()
#
#   class Keys:
#     EEG = 'EEG'
#     STAGES = 'STAGES'
#
#
#   # region: Public Mehtods
#
#   def report_detail(self):
#     console.show_info(f'{self.record_id}:')
#     for k, v in self.data_dict.items():
#       console.supplement(f'{k}.shape = {v.shape}')
#
#   # endregion: Public Mehtods
#
#   # region: Data IO
#   @classmethod
#   def load(cls, path):
#     sleep_data_list = cls.read_sleep_records(path)
#     preprocess_data_list = Data_Process.preprocess_data(sleep_data_list)
#     # print(len(preprocess_data_list[0].data_dict['EEG'][:, 1]))
#     data_set = cls.convert_to_dataset(preprocess_data_list)
#     train_size = int(data_set.data_dict['EEG'].shape[0] * 3 / 4)
#     validate_size = int(data_set.data_dict['EEG'].shape[0] * 1 / 10)
#     test_size = data_set.data_dict['EEG'].shape[0] - train_size - validate_size
#     print('^' * 30)
#     # print(data_set.data_dict['EEG'].shape[0])
#     print(train_size)
#     print(validate_size)
#     print(test_size)
#     print('^' * 30)
#     return cls._split_and_return(
#       data_set, train_size, validate_size, test_size, over_classes=False)
#     pass
#
#
#   @classmethod
#   def convert_to_dataset(cls, preprocess_data_list):
#     print('*' * 10)
#     print('preprocess_data_list:', len(preprocess_data_list))
#     sleep_eeg, sleep_stage = Data_Process.data_reshape(preprocess_data_list[0])
#     for sr_index in range(1, len(preprocess_data_list)):
#       sleep_eeg_temp, sleep_stage_temp = Data_Process.data_reshape(
#         preprocess_data_list[sr_index])
#       sleep_eeg = np.vstack(sleep_eeg, sleep_eeg_temp)
#       sleep_stage = np.vstack(sleep_stage, sleep_stage_temp)
#     data_set = SleepSet(sleep_eeg, sleep_stage, name=cls.DATA_NAME,
#                         **cls.PROPERTIES)
#     return data_set
#
#   @classmethod
#   def read_sleep_records(cls, data_dir: str):
#     # Sanity check
#     assert os.path.exists(data_dir)
#
#     # If .recs file exists, load it directly
#     folder_name = os.path.split(data_dir)[-1]
#     recs_fn = os.path.join(data_dir, folder_name + '-xai.recs')
#     if os.path.exists(recs_fn) and  os.stat(recs_fn).st_size != 0:
#       # with open support closing file in any situation
#       with open(recs_fn, 'rb') as file:
#         console.show_status('Loading `{}` ...'.format(recs_fn))
#         return pickle.load(file)
#
#     # Otherwise, read sleep records according to folder name
#     if folder_name in ['ucddb']:
#       sleep_records = cls._read_ucddb(data_dir)
#     elif folder_name in ['sleep-cassette', 'sleep-telemetry']:
#       sleep_records = cls._read_sleep_edf(data_dir)
#       pass
#     else:
#       raise KeyError(f'!! Unknown dataset `{folder_name}`')
#     # Save file and return
#     # with open(recs_fn, 'wb') as file:
#     #   pickle.dump(sleep_records, file, pickle.HIGHEST_PROTOCOL)
#     return sleep_records
#
#   # region: Builtin IO Methods
#
#   @classmethod
#   def _read_edf_data_file(cls, fn: str, channel_list: List[str]) -> np.ndarray:
#     from mne.io import concatenate_raws, read_raw_edf
#     from mne.io.edf.edf import RawEDF
#
#     with read_raw_edf(fn, preload=True) as raw_edf:
#       assert isinstance(raw_edf, RawEDF)
#       # edf_data[:, 0] is time, edf_data[:, k] is channel k (k >= 1)
#       edf_data = raw_edf.pick_channels(channel_list).to_data_frame().values
#     return edf_data
#
#   @classmethod
#   def _read_edf_anno_file(cls, fn: str):
#     from mne import read_annotations
#
#     print(fn)
#     stage_anno = []
#     # with read_annotations(fn) as raw_anno:
#     #   anno = raw_anno.to_data_frame().values
#     #   anno_dura = anno[:, 1]
#     #   anno_desc = anno[:, 2]
#     raw_anno = read_annotations(fn)
#     anno = raw_anno.to_data_frame().values
#     anno_dura = anno[:, 1]
#     anno_desc = anno[:, 2]
#     for dura_num in range(len(anno_dura) - 1):
#       for stage_num in range(int(anno_dura[dura_num]) // 30):
#         stage_anno.append(anno_desc[dura_num])
#     return np.array(stage_anno)
#
#   @classmethod
#   def _split_and_return(cls, data_set, train_size, validate_size, test_size,
#                         over_classes=False):
#     from tframe.data.dataset import DataSet
#     # assert isinstance(data_set, DataSet)
#     names, sizes = [], []
#     for name, size in zip(['Train Set', 'Validation Set', 'Test Set'],
#                           [train_size, validate_size, test_size]):
#       if size == 0: continue
#       names.append(name)
#       sizes.append(size)
#     data_sets = data_set.split(*sizes, over_classes=over_classes, names=names)
#     # Show data info
#     # cls._show_data_sets_info(data_sets)
#     return data_sets
#
#   @classmethod
#   def _read_ucddb(cls, data_dir: str):
#     # define static variables
#     class Channels:
#       EEG = {'siGNAl_EEG1': "C3A2",
#              'siGNAl_EEG2': "C4A1" }
#       ECG = {'siGNAL_ECG1': "chan 1",
#              'siGNAL_ECG2': "chan 2",
#              'siGNAL_ECG3': "chan 3",
#              'siGNAL_ECG4': "ECG"}
#
#     console.show_status('Reading `ucddb` dataset ...')
#
#     # Create an empty list
#     sleep_records: List[cls] = []
#
#     # Get all .edf files
#     rec_file_list: List[str] = walk(data_dir, 'file', '*.rec*')
#
#     # Read records one by one
#     for rec_fn in rec_file_list:
#       # Get id
#       id = os.path.split(rec_fn)[-1].split('.r')[0]
#       sr = SleepRecord()
#
#       console.show_status(f'Reading record `{id}` ...')
#       console.split()
#
#       # (1) Read EEG data
#       # Rename .rec file if necessary
#       if rec_fn[-3:] != 'edf':
#         os.rename(rec_fn, rec_fn + '.edf')
#         rec_fn = rec_fn + '.edf'
#
#       sr.data_dict['EEG'] = cls._read_edf_data_file(
#         rec_fn, list(Channels.EEG.values()))
#       console.split()
#
#       # (2) Read ECG data
#       fn = os.path.join(data_dir, id + '_lifecard.edf')
#       sr.data_dict['ECG'] = cls._read_edf_data_file(fn, list(Channels.ECG.values()))
#       console.split()
#
#       # (3) Read stage labels
#       fn = os.path.join(data_dir, id + '_stage.txt')
#       assert os.path.exists(fn)
#       with open(fn, 'r') as stage:
#         stage_ann = [line.strip() for line in stage.readlines()]
#       sr.data_dict['stage'] = np.array(stage_ann)
#
#       # (4) Read misc data
#       fn = os.path.join(data_dir, id + '_respevt.txt')
#       assert os.path.exists(fn)
#       with open(fn, 'r') as resp:
#         resp_content = resp.readlines()
#         resp_time = [line.strip().split()[0] for line in resp_content if ':' in line]
#         resp_anno = [line.strip().split()[1] for line in resp_content if ':' in line]
#       sr.data_dict['resp'] = np.array([resp_time,resp_anno])
#       # Append this record to list
#       sleep_records.append(sr)
#
#     # Return records
#     return sleep_records
#
#   @classmethod
#   def _read_sleep_edf(cls, data_dir: str):
#     # Define static variables
#     class Channels:
#       EEG = {'SIGNAl_EEG1': "EEG Fpz-Cz",
#              'SIGNAl_EEG2': "EEG Pz-Oz" }
#     console.show_status('Reading `sleep_edf` dataset ...')
#
#     # Create an empty list
#     sleep_records: List[cls] = []
#
#     # Get all .edf files
#     Hypnogram_file_list: List[str] = walk(data_dir, 'file', '*Hypnogram*')
#
#     # Read records one by one
#     for Hypnogram_fn in Hypnogram_file_list:
#       # Get id
#       id = os.path.split(Hypnogram_fn)[-1].split('-')[0]
#       sr = SleepRecord()
#
#       console.show_status(f'Reading record `{id}` ...')
#       console.split()
#
#       # (1) Read label
#       # Rename label file if necessary
#       # if Hypnogram_fn[7] != '0':
#       #   os.rename(Hypnogram_fn, Hypnogram_fn[:7] + '0' + '-Hypnogram.edf')
#       #   id = Hypnogram_fn[:7] + '0'
#       #   Hypnogram_fn = id + '-Hypnogram.edf'
#
#       sr.data_dict['stage'] = cls._read_edf_anno_file(Hypnogram_fn)
#
#       # (2) Read stage labels
#       fn = os.path.join(data_dir, id[:7] + '0' + '-PSG.edf')
#       assert os.path.exists(fn)
#       sr.data_dict['EEG'] = cls._read_edf_data_file(
#         fn, list(Channels.EEG.values()))
#       console.split()
#
#
#       # Append this record to list
#       sleep_records.append(sr)
#
#     # Return records
#     return sleep_records
#
#   # endregion: Builtin IO Methods
#
#   # endregion: Data IO
#
#
# if __name__ == '__main__':
#   # Get data path
#   abs_path = os.path.abspath(__file__)
#   data_dir = os.path.join(os.path.dirname(os.path.dirname(abs_path)),
#                           'data', 'sleep_edf', 'sleep-cassette')
#
#   #
#   # sleep_records = SleepRecord.read_sleep_records(data_dir)
#   # for slp_rec in sleep_records: slp_rec.report_detail()
#   SleepRecord.load(data_dir)
#

