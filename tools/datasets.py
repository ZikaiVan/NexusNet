from typing import List, Union
import sys, os
import torch
import numpy as np
import mne
from mne.filter import resample
from torch.utils.data import Dataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import Preprocessor
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.windowers import create_windows_from_events

class AttrDict(dict):
    def __init__(self, *config, **kwconfig):
        super(AttrDict, self).__init__(*config, **kwconfig)
        self.__dict__ = self
        for key in self:
            if type(self[key]) == dict:
                self[key] = AttrDict(self[key])

    def __getattr__(self, item):
        return None

    def get_values(self, keys):
        return {key: self.get(key) for key in keys}

    def dict(self):
        dictionary = dict(self)
        for key in dictionary:
            if type(dictionary[key]).__name__ == 'AttrDict':
                dictionary[key] = dict(dictionary[key])
        return dictionary
    

def print_off():
    sys.stdout = open(os.devnull, 'w')


def print_on():
    sys.stdout = sys.__stdout__


def get_edge_weight_from_electrode(edge_pos, delta=2):
    edge_pos_value=[v for k, v in edge_pos.items()]
    num_nodes=len(edge_pos_value)
    edge_weight = np.zeros([num_nodes, num_nodes])
    delta = delta
    edge_index = [[], []]

    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index[0].append(i)
            edge_index[1].append(j)
            if i == j:
                edge_weight[i][j] = 0
            else:
                edge_weight[i][j] = np.sum(
                    [(edge_pos_value[i][k] - edge_pos_value[j][k])**2 for k in range(2)])

    return edge_index, edge_weight


class BNCI20142a(Dataset):
    def __init__(
            self,
            subject: Union[int, List] = None,
            preproces_params: Union[dict, AttrDict] = None,
            phase: str = 'train',
            get_ch_names: bool = False
    ):
        if preproces_params is None:
            preproces_params = dict(
                band=[[0, 40]],
                start_time=-0.5
            )

        x_bundle, y_bundle = [], []
        for (low_hz, high_hz) in preproces_params['band']:
            x_list = []
            y_list = []

            if isinstance(subject, int):
                subject = [subject]

            print_off()

            # Load data from MOABBDataset
            dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject)

            # Preprocess data
            factor_new = 1e-3
            init_block_size = 1000
            preprocessors = [
                # Keep only EEG sensors
                Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False, apply_on_array=True),
                # Convert from volt to microvolt
                Preprocessor(fn=lambda x: x * 1e+06, apply_on_array=True),
                # Apply bandpass filtering
                Preprocessor(fn='filter', l_freq=low_hz, h_freq=high_hz, apply_on_array=True),
                # Apply exponential moving standardization
                Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new,
                             init_block_size=init_block_size, apply_on_array=True)
            ]
            preprocess(dataset, preprocessors)

            # Check sampling frequency
            sfreq = dataset.datasets[0].raw.info['sfreq']
            if not all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]):
                raise ValueError("Not match sampling rate.")

            # Divide data by trial
            trial_start_offset_samples = int(preproces_params['start_time'] * sfreq)

            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=trial_start_offset_samples,
                trial_stop_offset_samples=0,
                preload=True
            )
            print_on()

            # Make session-to-session data (subject dependent)
            if phase == 'train':
                for trial in windows_dataset.split('session')['session_T']:
                    x_list.append(trial[0])
                    y_list.append(trial[1])
            else:
                for trial in windows_dataset.split('session')['session_E']:
                    x_list.append(trial[0])
                    y_list.append(trial[1])

            # Return numpy array
            x_list = np.array(x_list)
            y_list = np.array(y_list)

            # Cut time points
            if preproces_params['end_time'] is not None:
                len_time = preproces_params['end_time'] - preproces_params['start_time']
                x_list = x_list[..., : int(len_time * sfreq)]

            # Resampling
            if preproces_params['resampling'] is not None:
                x_list = resample(np.array(x_list, dtype=np.float64), preproces_params['resampling'] / sfreq)

            x_bundle.append(x_list)
            y_bundle.append(y_list)

        self.x = np.stack(x_bundle, axis=1)
        self.y = np.array(y_bundle[0])
        #self.ch = dataset.

        if get_ch_names:
            self.ch_names = windows_dataset.datasets[0].windows.info.ch_names
        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = [self.x[idx], self.y[idx]]
        return sample
    
class BNCI20142b(Dataset):
    def __init__(
            self,
            subject: Union[int, List] = None,
            preproces_params: Union[dict, AttrDict] = None,
            phase: str = 'train',
            get_ch_names: bool = False
    ):
        if preproces_params is None:
            preproces_params = AttrDict(
                band=[[0, 40]],
                start_time=-0.5
            )

        x_bundle, y_bundle = [], []
        for (low_hz, high_hz) in preproces_params['band']:
            x_list = []
            y_list = []

            if isinstance(subject, int):
                subject = [subject]

            print_off()

            # Load data from MOABBDataset
            dataset = MOABBDataset(dataset_name="BNCI2014004", subject_ids=subject)
            # Preprocess data
            factor_new = 1e-3
            init_block_size = 1000

            preprocessors = [
                # Keep only EEG sensors
                Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False, apply_on_array=True),
                # Convert from volt to microvolt
                Preprocessor(fn=lambda x: x * 1e+06, apply_on_array=True),
                # Apply bandpass filtering
                Preprocessor(fn='filter', l_freq=low_hz, h_freq=high_hz, apply_on_array=True),
                # Apply exponential moving standardization
                Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new,
                             init_block_size=init_block_size, apply_on_array=True)
            ]
            preprocess(dataset, preprocessors)

            # Check sampling frequency
            sfreq = dataset.datasets[0].raw.info['sfreq']
            if not all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]):
                raise ValueError("Not match sampling rate.")

            # Divide data by trial
            trial_start_offset_samples = int(preproces_params['start_time'] * sfreq)

            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=trial_start_offset_samples,
                trial_stop_offset_samples=0,
                preload=True
            )
            print_on()

            # Make session-to-session data (subject dependent)
            if phase == 'train':
                for session in ['session_0', 'session_1', 'session_2']:
                    for trial in windows_dataset.split('session')[session]:
                        x_list.append(trial[0])
                        y_list.append(trial[1])
            else:
                for session in ['session_3', 'session_4']:
                    for trial in windows_dataset.split('session')[session]:
                        x_list.append(trial[0])
                        y_list.append(trial[1])

            # Return numpy array
            x_list = np.array(x_list)
            y_list = np.array(y_list)

            # Cut time points
            if preproces_params['end_time'] is not None:
                len_time = preproces_params['end_time'] - preproces_params['start_time']
                x_list = x_list[..., : int(len_time * sfreq)]

            # Resampling
            if preproces_params['resampling'] is not None:
                x_list = resample(np.array(x_list, dtype=np.float64), preproces_params['resampling'] / sfreq)

            x_bundle.append(x_list)
            y_bundle.append(y_list)

        self.x = np.stack(x_bundle, axis=1)
        self.y = np.array(y_bundle[0])

        if get_ch_names:
            self.ch_names = windows_dataset.datasets[0].windows.info.ch_names

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = [self.x[idx], self.y[idx]]
        return sample

def load_single_subject(dataset, subject_id, duration, to_tensor=True):
    tmp = sys.stdout
    if dataset=='BNCI2014001':
        trainset = BNCI20142a(subject=subject_id, 
            preproces_params={'start_time': -0.5,
                'end_time': duration,
                'band': [[0, 40]],
                'resampling': 250},
            phase='train', get_ch_names=True,
        )
        testset = BNCI20142a(subject=subject_id, 
                preproces_params={'start_time': -0.5,
                    'end_time': duration,
                    'band': [[0, 40]],
                    'resampling': 250},
                phase='test', get_ch_names=True,
            )
    elif dataset=='BNCI2014004':
        trainset = BNCI20142b(subject=subject_id, 
            preproces_params={'start_time': -0.5,
                'end_time': duration,
                'band': [[0, 40]],
                'resampling': 250},
            phase='train', get_ch_names=True,
        )
        testset = BNCI20142b(subject=subject_id, 
                preproces_params={'start_time': -0.5,
                    'end_time': duration,
                    'band': [[0, 40]],
                    'resampling': 250},
                phase='test', get_ch_names=True,
            ) 
    sys.stdout = tmp

    train_X = np.squeeze(trainset.x)
    train_y = np.squeeze(trainset.y)
    test_X = np.squeeze(testset.x)
    test_y = np.squeeze(testset.y)

    montage = mne.channels.make_standard_montage('biosemi64')
    all_pos = montage.get_positions()['ch_pos']
    electrodes_pos = {ch: all_pos[ch][:2]*100 for ch in trainset.ch_names}
    _, eu_adj = get_edge_weight_from_electrode(edge_pos=electrodes_pos)

    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y, eu_adj


