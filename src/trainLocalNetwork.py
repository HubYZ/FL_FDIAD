import copy
import os
import sys
import yaml
import time
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import multiprocessing
from utils import get_logger, get_gpus_memory_info

sys.path.append('..')
# from powerDataset import PowerDataLoader_load_every_time as PowerDataset
from powerDataset import PowerDataset
from models import PowerFDNet
from utils import FDIMetric

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 20)
np.set_printoptions(precision=4, linewidth=300)


def get_global_model(_network):
    return copy.deepcopy(_network.model)


class TrainLocalNetwork(object):

    def __init__(self, args_FL, subgrid):
        self.args = args_FL
        self.subgrid_data_dir = subgrid
        self._init_configure()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self._check_resume()

    def _init_configure(self):
        # parser = argparse.ArgumentParser(description='Process the command line.')
        # parser.add_argument('--col_bus', type=int, default=3)
        # parser.add_argument('--col_line', type=int, default=3)
        # parser.add_argument('--in_channel', type=int, default=1)
        # parser.add_argument('--channel_axis', type=int, default=1)
        # parser.add_argument('--num_classes', type=int, default=1)
        # parser.add_argument('--gpu_num', type=int, default=4)
        # parser.add_argument('--seq_size', type=int, default=2)
        # parser.add_argument('--report_freq', type=int, default=10)
        # parser.add_argument('--num_workers', type=int, default=1)
        # parser.add_argument('--prefetch_factor', type=int, default=1)
        # parser.add_argument('--grad_clip', type=float, default=5.0)
        # parser.add_argument('--learning_rate', type=float, default=0.001)

        # self.args = parser.parse_args()

        self.args.subgrid_data_dir = os.path.join(self.args.data_dir, self.subgrid_data_dir)
        self.args.epoch = self.args.local_ep
        self.args.batch_size = self.args.local_bs

        self.args.train_data_dir = os.path.join(self.args.subgrid_data_dir)
        self.args.val_data_dir_type_I = os.path.join(self.args.subgrid_data_dir, 'Type-I')
        self.args.val_data_dir_type_II = os.path.join(self.args.subgrid_data_dir, 'Type-II')
        self.args.val_data_dir_type_III = os.path.join(self.args.subgrid_data_dir, 'Type-III')

    def _init_device(self):
        if not torch.cuda.is_available():
            print('no gpu device available')
            sys.exit(1)

        np.random.seed(1337)
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        cudnn.enabled = True
        cudnn.benchmark = True
        self.device_id, self.gpus_info = get_gpus_memory_info()
        self.multi_GPU = True if self.args.gpu_num > 0 else False
        self.device = torch.device('cuda:{}'.format(0 if self.multi_GPU else self.device_id))


    def _init_dataset(self):
        self.channel_axis = self.args.channel_axis
        self.num_classes = self.args.num_classes
        self.seq_size = self.args.seq_size
        self.batch_size = self.args.batch_size

        num_workers = self.args.num_workers
        prefetch_factor = self.args.prefetch_factor

        dataset_train = PowerDataset(self.args, self.args.train_data_dir, train_flag='train')
        self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True, prefetch_factor=prefetch_factor,
                                           persistent_workers = True)

        dataset_val_type_I = PowerDataset(self.args, self.args.val_data_dir_type_I, train_flag='test')
        self.dataloader_val_I = DataLoader(dataset_val_type_I, batch_size=self.batch_size, shuffle=True,
                                         num_workers=num_workers, drop_last=True, prefetch_factor=prefetch_factor,
                                         persistent_workers = True)

        dataset_val_type_II = PowerDataset(self.args, self.args.val_data_dir_type_II, train_flag='test')
        self.dataloader_val_II = DataLoader(dataset_val_type_II, batch_size=self.batch_size, shuffle=True,
                                         num_workers=num_workers, drop_last=True, prefetch_factor=prefetch_factor,
                                         persistent_workers = True)

        dataset_val_type_III = PowerDataset(self.args, self.args.val_data_dir_type_III, train_flag='test')
        self.dataloader_val_III = DataLoader(dataset_val_type_III, batch_size=self.batch_size, shuffle=True,
                                         num_workers=num_workers, drop_last=True, prefetch_factor=prefetch_factor,
                                         persistent_workers = True)
        # persistent_workers = True

        self.line_num = dataset_train.num_line
        self.bus_num = dataset_train.num_bus
        self.line_col = dataset_train.col_line
        self.bus_col = dataset_train.col_bus
        self.in_channel = dataset_train.in_channel

    def _init_model(self):

        # Setup loss function
        self.lossBCE = nn.BCELoss() #.to(self.device)  # the input is the output of sigmoid

        model = PowerFDNet(line_num=self.line_num, bus_num=self.bus_num,
                           line_col=self.line_col, bus_col=self.bus_col,
                           num_classes=self.num_classes, channel_axis=self.channel_axis,
                           seq_size=self.seq_size, batch_size=self.batch_size,
                           in_channel=self.in_channel)

        if torch.cuda.device_count() > 1 and self.multi_GPU:
            model = nn.DataParallel(model)
        else:
            torch.cuda.set_device(self.device_id)

        self.model = model.to(self.device)

    def _check_resume(self):
        self.dur_time = 0
        self.start_epoch = 0
        self.best_loss = 1000000

        self.val_metric = FDIMetric()
        self.train_metric = FDIMetric()

    def run_train(self, model, global_round):
        epoch_loss = []

        self.model = model
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        for epoch in range(self.start_epoch, self.args.epoch):
            cur_epoch_loss = self.train(epoch)
            epoch_loss.append(cur_epoch_loss)
        # self.writer.close()
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train(self, epoch):
        self.model.train()

        batch_loss = []
        for step, cur_batch in enumerate(self.dataloader_train):

            bus_measurement = cur_batch['bus_meas']
            line_measurement = cur_batch['line_meas']
            label_measurement = cur_batch['label']

            bus_measurement = bus_measurement.view(self.batch_size*self.seq_size,1,self.bus_num,self.bus_col)
            line_measurement = line_measurement.view(self.batch_size*self.seq_size,1,self.line_num,self.line_col)
            label_measurement = label_measurement.view(self.batch_size,1)

            bus_measurement = bus_measurement.to(self.device)
            line_measurement = line_measurement.to(self.device)
            label_target = label_measurement.to(self.device)

            self.model_optimizer.zero_grad()

            binary_sigmoid = self.model(bus_measurement, line_measurement)

            # loss_tra
            train_loss = self.lossBCE(binary_sigmoid, label_target)

            train_loss.backward()

            if self.args.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.grad_clip)
            # Update the network parameters
            self.model_optimizer.step()

            batch_loss.append(train_loss.item())
        return sum(batch_loss)/len(batch_loss)

    def run_test(self, model):

        self.model = model

        val_loss_I = self.val_type_I()
        val_loss_II = self.val_type_II()
        val_loss_III, prec, recall, F1 = self.val_type_III()
        val_loss = (val_loss_I+val_loss_II+val_loss_III)/3.0

        return val_loss, prec, recall, F1

    def val_type_I(self):
        self.model.eval()

        with torch.no_grad():
            for step, cur_batch in enumerate(self.dataloader_val_I):
                bus_measurement = cur_batch['bus_meas']
                line_measurement = cur_batch['line_meas']
                label_measurement = cur_batch['label']

                bus_measurement = bus_measurement.view(self.batch_size * self.seq_size, 1, self.bus_num, self.bus_col)
                line_measurement = line_measurement.view(self.batch_size * self.seq_size, 1, self.line_num, self.line_col)
                label_measurement = label_measurement.view(self.batch_size, 1)

                bus_measurement = bus_measurement.to(self.device)
                line_measurement = line_measurement.to(self.device)
                label_target = label_measurement.to(self.device)

                binary_sigmoid = self.model(bus_measurement, line_measurement)

                # loss_tra
                val_loss = self.lossBCE(binary_sigmoid, label_target)

                self.val_metric.update(label_target.detach().cpu(), binary_sigmoid.detach().cpu())

            prec, recall, F1 = self.val_metric.get()

            #self.val_metric.reset()
            return val_loss

    def val_type_II(self):
        self.model.eval()

        with torch.no_grad():
            for step, cur_batch in enumerate(self.dataloader_val_II):
                bus_measurement = cur_batch['bus_meas']
                line_measurement = cur_batch['line_meas']
                label_measurement = cur_batch['label']

                bus_measurement = bus_measurement.view(self.batch_size * self.seq_size, 1, self.bus_num, self.bus_col)
                line_measurement = line_measurement.view(self.batch_size * self.seq_size, 1, self.line_num, self.line_col)
                label_measurement = label_measurement.view(self.batch_size, 1)

                bus_measurement = bus_measurement.to(self.device)
                line_measurement = line_measurement.to(self.device)
                label_target = label_measurement.to(self.device)

                binary_sigmoid = self.model(bus_measurement, line_measurement)

                # loss_tra
                val_loss = self.lossBCE(binary_sigmoid, label_target)

                self.val_metric.update(label_target.detach().cpu(), binary_sigmoid.detach().cpu())

            prec, recall, F1 = self.val_metric.get()

            #self.val_metric.reset()
            return val_loss

    def val_type_III(self):
        self.model.eval()

        with torch.no_grad():
            for step, cur_batch in enumerate(self.dataloader_val_III):
                bus_measurement = cur_batch['bus_meas']
                line_measurement = cur_batch['line_meas']
                label_measurement = cur_batch['label']

                bus_measurement = bus_measurement.view(self.batch_size * self.seq_size, 1, self.bus_num, self.bus_col)
                line_measurement = line_measurement.view(self.batch_size * self.seq_size, 1, self.line_num, self.line_col)
                label_measurement = label_measurement.view(self.batch_size, 1)

                bus_measurement = bus_measurement.to(self.device)
                line_measurement = line_measurement.to(self.device)
                label_target = label_measurement.to(self.device)

                binary_sigmoid = self.model(bus_measurement, line_measurement)

                # loss_tra
                val_loss = self.lossBCE(binary_sigmoid, label_target)

                self.val_metric.update(label_target.detach().cpu(), binary_sigmoid.detach().cpu())

            prec, recall, F1 = self.val_metric.get()

            self.val_metric.reset()
            return val_loss, prec, recall, F1


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    train_network = TrainLocalNetwork()
    train_network.run()
