from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pandas as pd
import pickle


class PowerDataset(Dataset):
    """
    Frank:
    suppose batch_size = 100, seq_size=13
    this is used to generate bus/line data with size of (100*13, 1, num_bus, col_bus)
    this data contains 100/2 normal data, 100/4 attack_va data, and 100/4 attack_vm data
    """
    def __init__(self, args, data_dir):

        self.batch_size = args.batch_size
        self.seq_size = args.seq_size

        self.normal_batch_size = np.around(self.batch_size * 0.5).astype(np.int)
        self.va_batch_size = np.around(self.batch_size * 0.25).astype(np.int)
        self.vm_batch_size = (self.batch_size - self.normal_batch_size - self.va_batch_size).astype(np.int)

        self.data_partition = 0.5

        self.in_channel = args.in_channel

        self.col_bus = args.col_bus  # bus feature
        self.col_line = args.col_line  # line feature

        self.attack_type = ['normal', 'normal', 'va_attack', 'vm_attack'] #defined in line 684 in SimBenchFDI.py
        self.num_measurement_type = 3 # attack type: Type-I Type-II Type-III

        self.saved_file_name = 'all_data.pkl'
        self.loadDataMeta(data_dir)
        self.getIndPoolValidBatch()

    def __len__(self):
        return len(self.ind_pool_valid_batch)

    def __getitem__(self, idx):
        batch_ind = self.ind_pool_valid_batch[idx]
        _bus_measurement, _line_measurement, _label_measurement = self.load_csv_to_ram(batch_ind)
        sample = {'bus_meas': _bus_measurement, 'line_meas': _line_measurement, 'label': _label_measurement}
        return sample

    def getIndPoolValidBatch(self):
        ind_pool_valid_batch = np.array([ind if (self._file_ok[ind-self.seq_size:ind]).sum() == self.seq_size else -1 for ind in range(self.seq_size, self.num_files)])
        self.ind_pool_valid_batch = ind_pool_valid_batch[ind_pool_valid_batch != -1]

    def loadDataMeta(self, data_dir):
        saved_file_path = data_dir+ '_' + self.saved_file_name
        if os.path.exists(saved_file_path):
            with open(saved_file_path,'rb') as f:
                self.num_files, self.num_bus, self.num_line, self.all_bus_meas, self.all_line_meas, self.meas_index, self._file_ok = pickle.load(f)
        else:
            self.loadAndSaveDataMeta(data_dir)

    def loadAndSaveDataMeta(self, data_dir):

        file_list = os.listdir(data_dir)

        self.num_files = 110000 #130000 # used to store file statue, should be large than the biggest file name

        # get num_bus and num_line
        _file_path = os.path.join(data_dir, file_list[0])
        meas = pd.read_csv(_file_path)
        attack_type = self.attack_type[0]
        _cur_bus_meas = meas[attack_type].loc[meas.element_type == 'bus']
        _cur_bus_meas = _cur_bus_meas.values.reshape(-1, self.col_bus)
        self.num_bus = _cur_bus_meas.shape[0]
        _cur_line_meas = meas[attack_type].loc[(meas.element_type == 'line') | (meas.element_type == 'trafo')]
        _cur_line_meas = _cur_line_meas.values.reshape(-1, self.col_line)
        self.num_line = _cur_line_meas.shape[0]

        self.all_bus_meas = np.zeros((self.num_files, self.num_measurement_type, self.num_bus, self.col_bus), dtype=np.float32)
        self.all_line_meas = np.zeros((self.num_files, self.num_measurement_type, self.num_line, self.col_line), dtype=np.float32)

        self._file_ok = np.zeros((self.num_files,), dtype=np.bool)

        _meas_type = ['normal','va_attack','vm_attack']
        self.meas_index = {'normal':0,'va_attack':1,'vm_attack':2}

        for _file in file_list:
            file_index = int(_file.split('_', 1)[0])
            _file_path = os.path.join(data_dir, _file)
            self._file_ok[file_index] = True

            meas = pd.read_csv(_file_path)
            for _cur_meas_type in _meas_type:
                _cur_bus_meas = meas[_cur_meas_type].loc[meas.element_type == 'bus']
                self.all_bus_meas[file_index][self.meas_index[_cur_meas_type]] = _cur_bus_meas.values.reshape(-1, self.col_bus).astype(np.float32)

                _cur_line_meas = meas[attack_type].loc[(meas.element_type == 'line') | (meas.element_type == 'trafo')]
                self.all_line_meas[file_index][self.meas_index[_cur_meas_type]] = _cur_line_meas.values.reshape(-1, self.col_line).astype(np.float32)

        with open(data_dir+ '_' +self.saved_file_name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.num_files, self.num_bus, self.num_line, self.all_bus_meas, self.all_line_meas, self.meas_index, self._file_ok], f)

    def load_csv_to_ram(self, batch_ind):
        _bus_meas = np.zeros((self.seq_size, self.num_bus, self.col_bus), dtype=np.float32)
        _line_meas = np.zeros((self.seq_size, self.num_line, self.col_line), dtype=np.float32)

        label_v = np.random.randint(0,4) # 1-2:normal, 3:attack_vm 4:attack_va

        attack_type = self.attack_type[label_v]
        _label = 0 if label_v < 2 else 1
        _label = np.ones((1, 1), dtype=np.float32)*_label

        ind_file_pool = np.arange(batch_ind - self.seq_size, batch_ind)

        _bus_meas = self.all_bus_meas[ind_file_pool, self.meas_index[attack_type],:,:]
        _line_meas = self.all_line_meas[ind_file_pool, self.meas_index[attack_type],:,:]

        _bus_measurement = torch.from_numpy(_bus_meas)
        _line_measurement = torch.from_numpy(_line_meas)
        _label_measurement = torch.from_numpy(_label).view(1, 1)
        return _bus_measurement, _line_measurement, _label_measurement


class PowerDataLoader_load_every_time(Dataset):
    """
    Frank:
    suppose batch_size = 100, seq_size=13
    this is used to generate bus/line data with size of (100*13, 1, num_bus, col_bus)
    this data contains 100/2 normal data, 100/4 attack_va data, and 100/4 attack_vm data
    """
    def __init__(self, args, data_dir):

        self.batch_size = args.batch_size
        self.seq_size = args.seq_size

        self.normal_batch_size = np.around(self.batch_size * 0.5).astype(np.int)
        self.va_batch_size = np.around(self.batch_size * 0.25).astype(np.int)
        self.vm_batch_size = (self.batch_size - self.normal_batch_size - self.va_batch_size).astype(np.int)

        self.data_partition = 0.5
        # TODO: decided by dataset

        self.in_channel = args.in_channel

        self.num_files = args.num_files
        self.col_bus = args.col_bus  # bus feature
        self.col_line = args.col_line  # line feature

        self.attack_type = ['normal', 'normal', 'va_attack', 'vm_attack'] #defined in line 684 in SimBenchFDI.py

        self.file_list, self.file_ok = self.loadDataMeta(data_dir)
        self.ind_pool_valid_batch = self.getIndPoolValidBatch()

    def __len__(self):
        return len(self.ind_pool_valid_batch)

    def __getitem__(self, idx):
        batch_ind = self.ind_pool_valid_batch[idx]
        _bus_measurement, _line_measurement, _label_measurement = self.load_csv_to_ram(batch_ind)
        sample = {'bus_meas': _bus_measurement, 'line_meas': _line_measurement, 'label': _label_measurement}
        return sample

    def getIndPoolValidBatch(self):
        ind_pool_valid_batch = np.array([ind if (self.file_ok[ind - self.seq_size:ind]).sum() == self.seq_size else -1 for ind in range(self.seq_size, self.num_files)])
        ind_pool_valid_batch = ind_pool_valid_batch[ind_pool_valid_batch != -1]

        return ind_pool_valid_batch

    def loadDataMeta(self, data_dir):

        file_list = os.listdir(data_dir)

        _file_path = os.path.join(data_dir, file_list[0])
        meas = pd.read_csv(_file_path)
        attack_type = self.attack_type[0]
        _cur_bus_meas = meas[attack_type].loc[meas.element_type == 'bus']
        _cur_bus_meas = _cur_bus_meas.values.reshape(-1, self.col_bus)
        self.num_bus = _cur_bus_meas.shape[0]
        _cur_line_meas = meas[attack_type].loc[(meas.element_type == 'line') | (meas.element_type == 'trafo')]
        _cur_line_meas = _cur_line_meas.values.reshape(-1, self.col_line)
        self.num_line = _cur_line_meas.shape[0]

        num_files = self.num_files
        _file_ok = np.zeros((num_files,), dtype=np.bool)
        _file_list = [None]*num_files

        for _file in file_list:
            file_index = int(_file.split('_', 1)[0])
            _file_path = os.path.join(data_dir, _file)
            _file_ok[file_index] = True
            _file_list[file_index] = _file_path

        return _file_list, _file_ok

    def load_csv_to_ram(self, batch_ind):
        _bus_meas = np.zeros((self.seq_size, self.num_bus, self.col_bus), dtype=np.float32)
        _line_meas = np.zeros((self.seq_size, self.num_line, self.col_line), dtype=np.float32)

        label_v = np.random.randint(0,4) # 1-2:normal, 3:attack_vm 4:attack_va

        attack_type = self.attack_type[label_v]
        _label = 0 if label_v < 2 else 1
        _label = np.ones((1, 1), dtype=np.float32)*_label

        ind_file_pool = range(batch_ind - self.seq_size, batch_ind)

        for ind, ind_file in enumerate(ind_file_pool):
            _file_path = self.file_list[ind_file]
            meas = pd.read_csv(_file_path)

            _cur_bus_meas = meas[attack_type].loc[meas.element_type == 'bus']
            _cur_bus_meas = _cur_bus_meas.values.reshape(-1, self.col_bus)
            # assert _cur_bus_meas.shape[0] == self.num_bus, "not enough bus data."

            _cur_line_meas = meas[attack_type].loc[(meas.element_type == 'line') | (meas.element_type == 'trafo')]
            _cur_line_meas = _cur_line_meas.values.reshape(-1, self.col_line)
            # assert _cur_line_meas.shape[0] == self.num_line, "not enough line data."

            _bus_meas[ind] = _cur_bus_meas
            _line_meas[ind] = _cur_line_meas

        _bus_measurement = torch.from_numpy(_bus_meas)
        _line_measurement = torch.from_numpy(_line_meas)
        _label_measurement = torch.from_numpy(_label).view(1, 1)
        return _bus_measurement, _line_measurement, _label_measurement


if __name__ == '__main__':
    pass
