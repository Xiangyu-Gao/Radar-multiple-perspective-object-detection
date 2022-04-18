import torch
import torch.nn as nn

from .BaseNet import BackboneNet, DetectNet
from .RFPose import RFPoseEncode, RFPoseDecode
from .CDC import RODEncode_RA, RODDecode_RA, RODEncode_RV, RODDecode_RV, RODEncode_VA, RODDecode_VA
from .Fuse import Fuse_fea, Fuse_fea_new, Fuse_fea_new_rep
from config import n_class, rodnet_configs, radar_configs


class RODNet(nn.Module):
    def __init__(self, n_class=n_class, win_size=rodnet_configs['win_size']):
        super(RODNet, self).__init__()
        self.backbone = BackboneNet()
        self.detect = DetectNet(n_class, win_size)
        # self.encode = RFPoseEncode()
        # self.decode = RFPoseDecode()
        self.c3d_encode_ra = RODEncode_RA()
        self.c3d_decode_ra = RODDecode_RA()
        self.c3d_encode_rv = RODEncode_RV()
        self.c3d_decode_rv = RODDecode_RV()
        self.c3d_encode_va = RODEncode_VA()
        self.c3d_decode_va = RODDecode_VA()
        # self.fuse_fea = Fuse_fea()
        # self.fuse_fea = Fuse_fea_new()
        self.fuse_fea = Fuse_fea_new_rep()

    def forward(self, x_ra, x_rv, x_va):
        # x = self.encode(x)
        # dets = self.decode(x)
        x_ra = self.c3d_encode_ra(x_ra)
        feas_ra = self.c3d_decode_ra(x_ra)  # (B, 32, W/2, 128, 128)
        x_rv = self.c3d_encode_rv(x_rv)
        feas_rv = self.c3d_decode_rv(x_rv)  # (B, 32, W/2, 128, 128)
        x_va = self.c3d_encode_va(x_va)
        feas_va = self.c3d_decode_va(x_va)  # (B, 32, W/2, 128, 128)
        dets = self.fuse_fea(feas_ra, feas_rv, feas_va) # (B, 3, W/2, 128, 128)
        dets2 = self.fuse_fea(torch.zeros_like(feas_ra), feas_rv, feas_va) # (B, 3, W/2, 128, 128)

        return dets, dets2
