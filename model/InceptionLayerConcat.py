import torch.nn as nn
import torch


class InceptionLayerConcat(nn.Module):
    """
    Kernal size: for 2d kernal size, since the kernal size in temporal domain will be fixed
    """
    def __init__(self, kernal_size, in_channel, stride):
        super(InceptionLayerConcat, self).__init__()

        paddingX = kernal_size[0] // 2
        paddingY = kernal_size[1] // 2

        self.branch1 = nn.Conv3d(in_channels=in_channel, out_channels=32,
                                 kernel_size=(5, kernal_size[0], kernal_size[1]), stride=stride, padding=(2, paddingX, paddingY))
        self.branch2a = nn.Conv3d(in_channels=in_channel, out_channels=64,
                                  kernel_size=(5, kernal_size[0], kernal_size[1]), stride=(1, 1, 1), padding=(2, paddingX, paddingY))
        self.branch2b = nn.Conv3d(in_channels=64, out_channels=64,
                                  kernel_size=(9, kernal_size[0], kernal_size[1]), stride=stride, padding=(4, paddingX, paddingY))
        self.branch3a = nn.Conv3d(in_channels=in_channel, out_channels=64,
                                  kernel_size=(5, kernal_size[0], kernal_size[1]), stride=(1, 1, 1), padding=(2, paddingX, paddingY))
        self.branch3b = nn.Conv3d(in_channels=64, out_channels=64,
                                  kernel_size=(13, kernal_size[0], kernal_size[1]), stride=stride, padding=(6, paddingX, paddingY))
                                
    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)

        branch3 = self.branch3a(x)
        branch3 = self.branch3b(branch3)

        # print(branch1.shape, branch2.shape, branch3.shape, x.shape)

        return torch.cat((branch1, branch2, branch3), 1)


if __name__ == '__main__':
    testLayer = InceptionLayerConcat((3, 3), 3, (1, 1, 1)).cuda()

    testTensor = torch.rand((1, 3, 16, 128, 128)).cuda()
    print(testLayer(testTensor).shape)
