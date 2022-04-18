import torch.nn as nn

from .InceptionLayerConcat import InceptionLayerConcat

from config import n_class, rodnet_configs, radar_configs


class RadarStackedHourglass(nn.Module):

    def __init__(self, stacked_num=1):
        super(RadarStackedHourglass, self).__init__()
        self.stacked_num = stacked_num
        self.conv1a = nn.Conv3d(in_channels=2, out_channels=32,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=32, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1c = nn.Conv3d(in_channels=64, out_channels=160,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))

        # self.inception1 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=2, stride=(1, 1, 1))
        # self.inception2 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=64, stride=(1, 1, 1))

        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([RODEncode(), RODDecode(),
                                                 nn.Conv3d(in_channels=160, out_channels=n_class,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2)),
                                                 nn.Conv3d(in_channels=n_class, out_channels=160,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2))]))
        self.hourglass = nn.ModuleList(self.hourglass)
        self.relu = nn.ReLU()
        self.bn1a = nn.BatchNorm3d(num_features=32)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn1c = nn.BatchNorm3d(num_features=160)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.bn1c(self.conv1c(x)))

        out = []
        for i in range(self.stacked_num):
            x, x1, x2, x3 = self.hourglass[i][0](x)
            x = self.hourglass[i][1](x, x1, x2, x3)
            confmap = self.hourglass[i][2](x)
            out.append(self.sigmoid(confmap))
            if i < self.stacked_num - 1:
                confmap_ = self.hourglass[i][3](confmap)
                x = x + confmap_

        return out


class RODEncode(nn.Module):

    def __init__(self):
        super(RODEncode, self).__init__()
        self.inception1 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.inception2 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.inception3 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))

        self.skip_inception1 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.skip_inception2 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.skip_inception3 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1 = nn.BatchNorm3d(num_features=160)
        self.bn2 = nn.BatchNorm3d(num_features=160)
        self.bn3 = nn.BatchNorm3d(num_features=160)

        self.skip_bn1 = nn.BatchNorm3d(num_features=160)
        self.skip_bn2 = nn.BatchNorm3d(num_features=160)
        self.skip_bn3 = nn.BatchNorm3d(num_features=160)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.skip_bn1(self.skip_inception1(x)))
        x = self.relu(self.bn1(self.inception1(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)

        x2 = self.relu(self.skip_bn2(self.skip_inception2(x)))
        x = self.relu(self.bn2(self.inception2(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)

        x3 = self.relu(self.skip_bn3(self.skip_inception3(x)))
        x = self.relu(self.bn3(self.inception3(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        # x = self.relu(self.bn5a(self.conv5a(x)))
        # x = self.relu(self.bn5b(self.conv5b(x)))

        return x, x1, x2, x3


class RODDecode(nn.Module):

    def __init__(self):
        super(RODDecode, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=160, out_channels=160,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=160, out_channels=160,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=160, out_channels=160,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.conv1 = nn.Conv3d(in_channels=160, out_channels=160,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=160, out_channels=160,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=160, out_channels=160,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.convt4 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
        #                                  kernel_size=(3, 6, 6), stride=(1, 4, 4), padding=(1, 1, 1))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
                                          radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x, x1, x2, x3):
        x = self.prelu(self.convt1(x + x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.convt2(x + x2))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.prelu(self.conv2(x))
        x = self.prelu(self.convt3(x + x1))  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        x = self.prelu(self.conv3(x))
        # x = self.convt4(x)
        # x = self.upsample(x)
        # x = self.sigmoid(x)

        return x
