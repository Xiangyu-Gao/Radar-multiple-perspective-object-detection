import torch
import torch.nn as nn

from .BaseNet import BackboneNet, DetectNet
from .RFPose import RFPoseEncode, RFPoseDecode
from .CDC import RODEncode, RODDecode
from .HG import RadarStackedHourglass

from config import n_class, rodnet_configs, radar_configs


class RODNet(nn.Module):
    def __init__(self, n_class=n_class, win_size=rodnet_configs['win_size'], stacked_num=2):
        super(RODNet, self).__init__()
        self.backbone = BackboneNet()
        self.detect = DetectNet(n_class, win_size)
        # self.encode = RFPoseEncode()
        # self.decode = RFPoseDecode()
        self.stacked_hourglass = RadarStackedHourglass(stacked_num=2)

    def forward(self, x):
        # x = self.encode(x)
        # dets = self.decode(x)
        out = self.stacked_hourglass(x)
        return out
