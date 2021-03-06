import torch
import torch.nn as nn

import time


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class PowerFDNet(nn.Module):
    """Construct a network"""

    def __init__(self, line_num=100, bus_num=100, line_col=1, bus_col=1, num_classes=2, channel_axis=1, seq_size=96, batch_size=50, in_channel=1):
        super(PowerFDNet, self).__init__()

        # the stem need a complicate mode
        self.in_channels = in_channel

        self.out_channels = 12
        self.out_linear = 4

        self.bus_col = bus_col
        self.line_col = line_col
        self.num_classes = num_classes
        self.input_size = line_num + bus_num
        self.channel_dim = channel_axis

        self.seq_size = seq_size
        # self.batch_size = batch_size

        # x_bus  = [50x96, 1, 50, 3]
        self.busConv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, self.bus_col)),
            nn.BatchNorm2d(self.out_channels),
            nn.ELU()
        )

        # x_bus = [50x96, 12, 50, 1]
        # x = x.permute(x) = [50x96, 1, 50, 12]
        self.busConv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_linear, kernel_size=(1, self.out_channels)),
            nn.BatchNorm2d(self.out_linear),
            nn.ELU()
        )
        # # x = x.permute(x) = [50x96, 1, 50, 4]
        # # x = [50x96, 1, 50, 4]

        # x_line = [50x96, 1, 60, 4]
        self.lineConv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, self.line_col)),
            nn.BatchNorm2d(self.out_channels),
            nn.ELU()
        )

        # x_line = [50x96, 12, 60, 1]
        # x = x.permute(x) = [50x96, 1, 60, 12]
        self.lineConv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_linear, kernel_size=(1, self.out_channels)),
            nn.BatchNorm2d(self.out_linear),
            nn.ELU()
        )
        # # x = x.permute(x) = [50x96, 1, 60, 4]
        # x = [50x96, 1, 60, 4]

        # x = torch.cat((x_bus, x_line), dim=2) = [50x96, 1, 110, 4]

        self.gridConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.input_size*2, kernel_size=(self.input_size, self.out_linear),),
            nn.BatchNorm2d(self.input_size*2),
            nn.ELU()
        )

        # x = [50x96, 220, 1, 1]

        # x=x.squeeze(x).reshape() = [50, 96, 220]

        self.layer1LSTM = nn.LSTM(self.input_size*2, self.input_size*2, 1, batch_first=True)
        # x = [50, 96, 220]

        self.layer2LSTM = nn.LSTM(self.input_size*2, self.input_size*4, 1, batch_first=True)
        # x = [50, 96, 440]

        # x = x[:,-1,:] = [50, 440]
        self.outLinear = nn.Linear(in_features=self.input_size*4, out_features=1)
        # x = [50, 1]

        self.outSigmoid = nn.Sigmoid()
        # x = [50, 1]

        self.apply(weights_init)

    def forward(self, x_bus, x_line):
        y_bus = self.busConv1(x_bus)
        y_bus = y_bus.permute(0, 3, 2, 1)
        y_bus = self.busConv2(y_bus)
        y_bus = y_bus.permute(0, 3, 2, 1)

        y_line = self.lineConv1(x_line)
        y_line = y_line.permute(0, 3, 2, 1)
        y_line = self.lineConv2(y_line)
        y_line = y_line.permute(0, 3, 2, 1)

        h_feature = torch.cat((y_bus, y_line), dim=self.channel_dim + 1)

        h_feature = self.gridConv(h_feature)

        h_feature = h_feature.squeeze()
        h_feature = h_feature.view(-1, self.seq_size, self.input_size*2)

        h_feature, _ = self.layer1LSTM(h_feature)
        h_feature, _ = self.layer2LSTM(h_feature)

        h_feature = h_feature[:, -1, :]
        h_feature = self.outLinear(h_feature)
        binary_sigmoid = self.outSigmoid(h_feature)

        # time.sleep(5)

        return binary_sigmoid


if __name__ == "__main__":
    x_bus = torch.rand([3*96,1,20,3])
    y_line = torch.rand([3*96,1,15,3])
    net = PowerFDNet(15, 20, 3,3,1,1,96,3,1)
    y = net(x_bus,y_line)
    pass
