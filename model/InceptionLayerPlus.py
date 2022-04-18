import torch.nn as nn
import torch

class InceptionLayerPlus(nn.Module):
    """
    Kernal size: for 2d kernal size, since the kernal size in temporal domain will be fixed
    """

    def __init__(self, kernal_size, in_channel, out_channel, stride):
        super(InceptionLayerPlus, self).__init__()

        paddingX = kernal_size[0] // 2
        paddingY = kernal_size[1] // 2

        self.branch3 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=(3, kernal_size[0], kernal_size[1]), stride=stride, padding=(1, paddingX, paddingY))
        self.branch7x7a = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=(3, kernal_size[0], kernal_size[1]), stride=(1, 1, 1), padding=(1, paddingX, paddingY))
        self.branch7x7b = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=(7, kernal_size[0], kernal_size[1]), stride=stride, padding=(3, paddingX, paddingY))
        self.branch9x9a = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=(3, kernal_size[0], kernal_size[1]), stride=(1, 1, 1), padding=(1, paddingX, paddingY))
        self.branch9x9b = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=(9, kernal_size[0], kernal_size[1]), stride=stride, padding=(4, paddingX, paddingY))
        
    def forward(self, x):
        branch3 = self.branch3(x)

        branch7 = self.branch7x7a(x)
        branch7 = self.branch7x7b(branch7)

        branch9 = self.branch9x9a(x)
        branch9 = self.branch9x9b(branch9)

        return branch9 + branch3 + branch7

if __name__ == '__main__':
    testLayer = InceptionLayerPlus((3, 3), 3, 256, (1, 1, 1)).cuda()

    testTensor = torch.rand((1, 3, 16, 128, 128)).cuda()
    print(testLayer(testTensor).shape)
