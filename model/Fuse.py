import torch.nn as nn
import torch
import torch.nn.functional as F
from config import n_class, rodnet_configs, radar_configs
n_angle = 128
n_range = 128
n_vel = 128
n_win = 16
n_rcsfea = 16


class Fuse_fea(nn.Module):
    def __init__(self):
        super(Fuse_fea, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=96, out_channels=48,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=48, out_channels=n_class,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va):
        feas_rv = torch.sum(feas_rv, 4, keepdim=True) # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_ra1 = feas_rv.expand(-1, -1, -1, -1, n_angle)
        feas_va = torch.sum(feas_va, 4, keepdim=True)  # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_va = torch.transpose(feas_va, 3, 4)    # (B, 32, W/2, 128, 1) -> (B, 32, W/2, 1, 128)
        feas_ra2 = feas_va.expand(-1, -1, -1, n_range, -1)  # (B, 32, W/2, 1, 128) -> (B, 32, W/2, 128, 128)
        feas_ra = torch.cat((feas_ra, feas_ra1, feas_ra2), 1) # 3*(B, 32, W/2, 128, 128) -> (B, 96, W/2, 128, 128)

        x = self.relu(self.convt1(feas_ra))  # (B, 96, W/2, 128, 128) -> (B, 48, W/2, 128, 128)
        x = self.sigmoid(self.convt2(x))  # (B, 48, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x


class Fuse_fea_new(nn.Module):
    def __init__(self):
        super(Fuse_fea_new, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=32, out_channels=32,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=32, out_channels=n_class,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va):
        feas_rv = torch.sum(feas_rv, 4, keepdim=True) # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_ra1 = feas_rv.expand(-1, -1, -1, -1, n_angle)
        feas_va = torch.sum(feas_va, 4, keepdim=True)  # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_va = torch.transpose(feas_va, 3, 4)    # (B, 32, W/2, 128, 1) -> (B, 32, W/2, 1, 128)
        feas_ra2 = feas_va.expand(-1, -1, -1, n_range, -1)  # (B, 32, W/2, 1, 128) -> (B, 32, W/2, 128, 128)

        fea_shap = feas_ra.shape # (B, 32, W/2, 128, 128)
        feas_ra = feas_ra.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra = torch.unsqueeze(torch.reshape(feas_ra, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra1 = feas_ra1.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra1 = torch.unsqueeze(torch.reshape(feas_ra1, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra2 = feas_ra2.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra2 = torch.unsqueeze(torch.reshape(feas_ra2, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra = torch.cat((feas_ra, feas_ra1, feas_ra2), 2) # 3*(B*W/2, 32, 1, 128, 128) -> (B*W/2, 32, 3, 128, 128)

        x = torch.squeeze(self.relu(self.convt1(feas_ra)))  # (B*W/2, 32, 3, 128, 128) -> (B*W/2, 32, 1, 128, 128) -> (B*W/2, 32, 128, 128)
        x = torch.transpose(torch.reshape(x, (fea_shap[0], fea_shap[2], fea_shap[1], fea_shap[3], fea_shap[4])), 1, 2) # (B*W/2, 32, 128, 128) -> (B, W/2, 32, 128, 128) -> (B, 32, W/2, 128, 128)
        x = self.sigmoid(self.convt2(x))  # (B, 32, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x


class Fuse_fea_new_rep(nn.Module):
    def __init__(self):
        super(Fuse_fea_new_rep, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.convt3 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 1, 21), stride=(1, 1, 1), padding=(0, 0, 0),
                                dilation=(1, 1, 6)) # padding 60
        self.convt4 = nn.Conv3d(in_channels=48, out_channels=n_class,
                                kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va):
        feas_rv = torch.sum(feas_rv, 4, keepdim=True) # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_ra1 = feas_rv.expand(-1, -1, -1, -1, n_angle)
        feas_va = torch.sum(feas_va, 4, keepdim=True)  # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_va = torch.transpose(feas_va, 3, 4)    # (B, 32, W/2, 128, 1) -> (B, 32, W/2, 1, 128)
        feas_ra2 = feas_va.expand(-1, -1, -1, n_range, -1)  # (B, 32, W/2, 1, 128) -> (B, 32, W/2, 128, 128)

        fea_shap = feas_ra.shape # (B, 32, W/2, 128, 128)
        feas_ra = feas_ra.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra = torch.unsqueeze(torch.reshape(feas_ra, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra1 = feas_ra1.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra1 = torch.unsqueeze(torch.reshape(feas_ra1, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra2 = feas_ra2.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra2 = torch.unsqueeze(torch.reshape(feas_ra2, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra = torch.cat((feas_ra, feas_ra1, feas_ra2), 2) # 3*(B*W/2, 32, 1, 128, 128) -> (B*W/2, 32, 3, 128, 128)

        x1 = torch.squeeze(self.prelu(self.convt1(feas_ra)))  # (B*W/2, 32, 3, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        x2 = torch.squeeze(self.prelu(self.convt2(feas_ra)))  # (B*W/2, 32, 3, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        feas_ra = F.pad(feas_ra, (60, 60, 0, 0, 0, 0), "circular")
        x3 = torch.squeeze(self.prelu(self.convt3(feas_ra)))  # (B*W/2, 32, 3, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        x1 = torch.cat((x1, x2, x3), 1) # (B*W/2, 16, 128, 128) -> (B*W/2, 48, 128, 128)

        x = torch.transpose(torch.reshape(x1, (fea_shap[0], fea_shap[2], 48, fea_shap[3], fea_shap[4])), 1, 2) # (B*W/2, 48, 128, 128) -> (B, W/2, 48, 128, 128) -> (B, 48, W/2, 128, 128)
        x = self.sigmoid(self.convt4(x))  # (B, 48, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x


class Fuse_fea_maxpool(nn.Module):
    def __init__(self):
        super(Fuse_fea_maxpool, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(2, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.convt3 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(2, 1, 21), stride=(1, 1, 1), padding=(0, 0, 0),
                                dilation=(1, 1, 6)) # padding 60
        self.convt4 = nn.Conv3d(in_channels=48, out_channels=n_class,
                                kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.maxpool = nn.MaxPool1d(128)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va):
        feas_rv = torch.unsqueeze(feas_rv, 4) # (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1, 128)
        feas_rv = feas_rv.expand(-1, -1, -1, -1, n_angle, -1) # (B, 32, W/2, 128, 1, 128) -> (B, 32, W/2, 128, 128, 128)
        feas_va = torch.unsqueeze(feas_va, 3) # (B, 32, W/2, 128, 128) -> (B, 32, W/2, 1, 128, 128)
        feas_va = feas_va.expand(-1, -1, -1, n_range, -1, -1)  # (B, 32, W/2, 1, 128, 128) -> (B, 32, W/2, 128, 128, 128)
        fea_shap = feas_ra.shape  # (B, 32, W/2, 128, 128)
        feas_ra1 = torch.max(feas_rv + feas_va, 5) # (B, 32, W/2, 128, 128)

        feas_ra = feas_ra.permute(0, 2, 1, 3, 4)  # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra = torch.unsqueeze(torch.reshape(feas_ra, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2)
                    # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra1 = feas_ra1.permute(0, 2, 1, 3, 4)  # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra1 = torch.unsqueeze(torch.reshape(feas_ra1, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2)
                    # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra = torch.cat((feas_ra, feas_ra1), 2) # (B*W/2, 32, 1, 128, 128) -> (B*W/2, 32, 2, 128, 128)

        x1 = torch.squeeze(self.prelu(self.convt1(feas_ra)))  # (B*W/2, 32, 2, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        x2 = torch.squeeze(self.prelu(self.convt2(feas_ra)))  # (B*W/2, 32, 2, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        feas_ra = F.pad(feas_ra, (60, 60, 0, 0, 0, 0), "circular")
        x3 = torch.squeeze(self.prelu(self.convt3(feas_ra)))  # (B*W/2, 32, 2, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        x1 = torch.cat((x1, x2, x3), 1) # (B*W/2, 16, 128, 128) -> (B*W/2, 48, 128, 128)

        x = torch.transpose(torch.reshape(x1, (fea_shap[0], fea_shap[2], 48, fea_shap[3], fea_shap[4])), 1, 2) # (B*W/2, 48, 128, 128) -> (B, W/2, 48, 128, 128) -> (B, 48, W/2, 128, 128)
        x = self.sigmoid(self.convt4(x))  # (B, 48, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x


class Fuse_fea_vsup(nn.Module):
    def __init__(self):
        super(Fuse_fea_vsup, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=96, out_channels=48,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=48, out_channels=n_class,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt3 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt4 = nn.Conv3d(in_channels=16, out_channels=n_class,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt5 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt6 = nn.Conv3d(in_channels=16, out_channels=n_class,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va):
        x_rv = self.prelu(self.convt3(feas_rv))
        x_rv = self.convt4(x_rv)
        x_va = self.prelu(self.convt5(feas_va))
        x_va = self.convt6(x_va)

        feas_rv = torch.sum(feas_rv, 4, keepdim=True) # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_ra1 = feas_rv.expand(-1, -1, -1, -1, n_angle)
        feas_va = torch.sum(feas_va, 4, keepdim=True)  # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_va = torch.transpose(feas_va, 3, 4)    # (B, 32, W/2, 128, 1) -> (B, 32, W/2, 1, 128)
        feas_ra2 = feas_va.expand(-1, -1, -1, n_range, -1)  # (B, 32, W/2, 1, 128) -> (B, 32, W/2, 128, 128)
        feas_ra = torch.cat((feas_ra, feas_ra1, feas_ra2), 1) # 3*(B, 32, W/2, 128, 128) -> (B, 96, W/2, 128, 128)

        x = self.prelu(self.convt1(feas_ra))  # (B, 96, W/2, 128, 128) -> (B, 48, W/2, 128, 128)
        x = self.sigmoid(self.convt2(x))  # (B, 48, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x, x_rv, x_va


class Fuse_fea_wrcs(nn.Module):
    def __init__(self):
        super(Fuse_fea_wrcs, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=112, out_channels=56,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=56, out_channels=n_class,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt3 = nn.Conv2d(in_channels=1, out_channels=4,
                                kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.convt4 = nn.Conv2d(in_channels=4, out_channels=n_rcsfea,
                                kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va, rcs_ra):
        rcs_ra = rcs_ra.permute(0, 2, 1, 3, 4)  # (B, 1, W/2, 128, 128) ->(B, W/2, 1, 128, 128)
        rcs_ra = self.relu(self.convt3(rcs_ra.view(-1, 1, n_range, n_angle)))   #(B, W/2, 1, 128, 128) -> (BW/2, 1, 128, 128) -> (BW/2, 4, 128, 128)
        rcs_ra = self.relu(self.convt4(rcs_ra)) # (BW/2, 4, 128, 128) -> (BW/2, 16, 128, 128)
        rcs_ra = rcs_ra.view(-1, n_win, n_rcsfea, n_range, n_angle).permute(0, 2, 1, 3, 4)    # (BW/2, 16, 128, 128) -> (B, W/2, 16, 128, 128) -> (B, 16, W/2, 128, 128)
        feas_rv = torch.sum(feas_rv, 4, keepdim=True)  # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_ra1 = feas_rv.expand(-1, -1, -1, -1, n_angle)
        feas_va = torch.sum(feas_va, 4, keepdim=True)  # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_va = torch.transpose(feas_va, 3, 4)  # (B, 32, W/2, 128, 1) -> (B, 32, W/2, 1, 128)
        feas_ra2 = feas_va.expand(-1, -1, -1, n_range, -1)  # (B, 32, W/2, 1, 128) -> (B, 32, W/2, 128, 128)
        feas_ra = torch.cat((feas_ra, feas_ra1, feas_ra2, rcs_ra), 1)  # 3*(B, 32, W/2, 128, 128) + (B, 16, W/2, 128, 128)-> (B, 112, W/2, 128, 128)

        x = self.relu(self.convt1(feas_ra))  # (B, 112, W/2, 128, 128) -> (B, 56, W/2, 128, 128)
        x = self.sigmoid(self.convt2(x))  # (B, 56, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x


class Fuse_fea_2ra(nn.Module):
    def __init__(self):
        super(Fuse_fea_2ra, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=64, out_channels=32,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=32, out_channels=n_class,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra1, feas_ra2):
        feas_ra = torch.cat((feas_ra1, feas_ra2), 1) # 2*(B, 32, W/2, 128, 128) -> (B, 64, W/2, 128, 128)
        x = self.prelu(self.convt1(feas_ra))  # (B, 64, W/2, 128, 128) -> (B, 32, W/2, 128, 128)
        x = self.convt2(x)  # (B, 32, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x