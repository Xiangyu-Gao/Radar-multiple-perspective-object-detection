import torch
import torch.nn as nn

from .BaseNet import BackboneNet, DetectNet
from .RFPose import RFPoseEncode, RFPoseDecode
from .CDC import RODEncode, RODDecode

from config import n_class, rodnet_configs, radar_configs


class RODNet(nn.Module):
    def __init__(self, n_class=n_class, win_size=rodnet_configs['win_size']):
        super(RODNet, self).__init__()
        self.backbone = BackboneNet()
        self.detect = DetectNet(n_class, win_size)
        # self.encode = RFPoseEncode()
        # self.decode = RFPoseDecode()
        self.c3d_encode = RODEncode()
        self.c3d_decode = RODDecode()

    def forward(self, x):
        # x = self.encode(x)
        # dets = self.decode(x)
        x = self.c3d_encode(x)
        dets = self.c3d_decode(x)
        return dets
