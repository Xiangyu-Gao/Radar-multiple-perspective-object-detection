import torch.nn as nn

from config import n_class, rodnet_configs, radar_configs


class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        return x


class DetectNet(nn.Module):
    def __init__(self, n_class=n_class, win_size=rodnet_configs['win_size']):
        super(DetectNet, self).__init__()
        self.win_size = win_size
        self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(16, n_class, (2, 2, 2), stride=(2, 2, 2))
        self.upsample = nn.Upsample(size=(self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                    mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        x = self.sigmoid(x)
        return x
