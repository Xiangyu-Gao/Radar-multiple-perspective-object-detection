import torch
import os
from model.RODNet_CDC2c_mid import RODNet

def main():
    # n_class = 3
    # win_size = 16
    # # load new model
    # rodnet = RODNet(n_class=n_class, win_size=win_size).cuda()
    # for name, param in rodnet.named_parameters():
    #     # if param.requires_grad:
    #     #    print(name, param.data)
    #     if 'c3d_encode' in name:
    #         # print(name)
    #         param.requires_grad = False
    #
    # input()
    # model_dict = rodnet.state_dict()
    # # for k, v in model_dict.items():
    # #     if k == 'c3d_encode_ra2.bn2b.weight':
    # #         a1 = v
    # #         print('has')
    # #         print(v)

    # # load replace model
    # replace_PATH = '/mnt/sda/results/CDC2c_mid-20200401-140457/rodnet_01_0000000001_000001.pkl'
    # replace_checkpoint = torch.load(replace_PATH)
    # model_dict = replace_checkpoint['model_state_dict']
    # # print(replace_checkpoint['optimizer_state_dict'])
    # # input()
    # # load pretrained model
    # pre_PATH = '/mnt/sda/results/CDC2c-20200327-013734/rodnet_30_0000029123_000001.pkl'
    # checkpoint = torch.load(pre_PATH)
    # pretrained_dict = checkpoint['model_state_dict']
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'c3d_encode' in k}
    # # for k, v in pretrained_dict.items():
    # #     if k == 'c3d_encode_ra2.bn2b.weight':
    # #         a2 = v
    # #         print(v)
    #
    # model_dict.update(pretrained_dict)
    # for k, v in model_dict.items():
    #     print(k)
    #     # if k == 'c3d_encode_ra2.bn2b.weight':
    #     #     a3 = v
    #     #     print(v)
    # #
    # # print(a1-a3)
    # # print(a2-a3)
    #
    # # model.load_state_dict(model_dict)
    #
    # # save model
    # status_dict = {
    #     'epoch': replace_checkpoint['epoch'],
    #     'iter': replace_checkpoint['iter'],
    #     'model_state_dict': model_dict,
    #     'optimizer_state_dict': replace_checkpoint['optimizer_state_dict'],
    #     'loss': replace_checkpoint['loss'],
    #     'iter_count': replace_checkpoint['iter_count'],
    # }
    # save_model_path = os.path.join('/mnt/sda/results/CDC2c_mid-20200401-140457/rodnet_01_0000000001_000002.pkl')
    # torch.save(status_dict, save_model_path)

    # # load replace model
    # replace_PATH = '/mnt/sda/results/C3D_vsup-20200404-181324/rodnet_01_0000000001_000001.pkl'
    # replace_checkpoint = torch.load(replace_PATH)
    # model_dict = replace_checkpoint['model_state_dict']
    # print(replace_checkpoint['optimizer_state_dict'])
    # for k, v in model_dict.items():
    #     print(k)
    # input()
    # for k, v in model_dict.items():
    #     if k == 'c3d_decode_va.convt2.weight':
    #         a1 = v
    #         # print(v)
    # # load pretrained model
    # pre_PATH = '/mnt/sda/results/C3D-20200304-212224/rodnet_36_0000034122_000001.pkl'
    # checkpoint = torch.load(pre_PATH)
    # pretrained_dict = checkpoint['model_state_dict']
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # for k, v in pretrained_dict.items():
    # #     print(k)
    # # input()
    #
    # for k, v in pretrained_dict.items():
    #     if k == 'c3d_decode_va.convt2.weight':
    #         a2 = v
    #         # print(v)
    # model_dict.update(pretrained_dict)
    # for k, v in model_dict.items():
    #     if k == 'c3d_decode_va.convt2.weight':
    #         a3 = v
    #         # print(v)
    #
    # print(a1-a3)
    # print(a2-a3)
    # #
    # # # model.load_state_dict(model_dict)
    # #
    # # save model
    # status_dict = {
    #     'epoch': replace_checkpoint['epoch'],
    #     'iter': replace_checkpoint['iter'],
    #     'model_state_dict': model_dict,
    #     'optimizer_state_dict': replace_checkpoint['optimizer_state_dict'],
    #     'loss': replace_checkpoint['loss'],
    #     'iter_count': replace_checkpoint['iter_count'],
    # }
    # save_model_path = os.path.join('/mnt/sda/results/C3D_vsup-20200404-181324/rodnet_01_0000000001_000002.pkl')
    # torch.save(status_dict, save_model_path)

    # load replace model
    replace_PATH = '/mnt/sda/results/CDC2c_Cpx-20200513-165757/rodnet_01_0000000001_000002.pkl'
    replace_checkpoint = torch.load(replace_PATH)
    model_dict = replace_checkpoint['model_state_dict']

    # for k, v in model_dict.items():
    #     print(k)
    #     if k == 'fuse_fea.convt1.weight':
    #         print(v[:,:, 0, 0, 0])
    # input()

    # load pretrained model
    pre_PATH = '/mnt/sda/results/CDC2c_Cpx-20200512-221542/rodnet_09_0000011608_000001.pkl'
    checkpoint = torch.load(pre_PATH)
    pretrained_dict = checkpoint['model_state_dict']
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
    #                    and ('fuse_fea.convt4' not in k)}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    for k, v in pretrained_dict.items():
        print(k)
    input()

    # for k, v in pretrained_dict.items():
    #     if k == 'c3d_decode_va.convt2.weight':
    #         a2 = v
    #         # print(v)
    model_dict.update(pretrained_dict)
    # for k, v in model_dict.items():
    #     if k == 'c3d_decode_va.convt2.weight':
    #         a3 = v
    #         # print(v)
    #
    # print(a1 - a3)
    # print(a2 - a3)
    #
    # # model.load_state_dict(model_dict)
    # save model
    status_dict = {
        'epoch': replace_checkpoint['epoch'],
        'iter': replace_checkpoint['iter'],
        'model_state_dict': model_dict,
        'optimizer_state_dict': replace_checkpoint['optimizer_state_dict'],
        'loss': replace_checkpoint['loss'],
        'iter_count': replace_checkpoint['iter_count'],
    }
    save_model_path = os.path.join('/mnt/sda/results/CDC2c_Cpx-20200513-165757/rodnet_01_0000000001_000003.pkl')
    torch.save(status_dict, save_model_path)







if __name__ == '__main__':
    main()