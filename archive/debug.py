import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from utils.read_annotations import read_ra_labels_csv
from utils.mappings import confmap2ra
from config import radar_configs
# from model.RODNet_CDCsc import RODNet
from model.RODNet_CDC import RODNet
# from model.RODNet_HG import RODNet

# with open('./data/confmaps_gt/train_od/2019_09_29_onrd002.pkl', 'rb') as f:
#     data = pickle.load(f)
#     seq = 700
#     print(len(data[0]))
#     print(len(data[1]))
#     print(data[0][seq].shape)
#     print(data[1][seq])
#     raw = data[0][seq]
#     inds = data[1][seq]
#     for i in inds:
#         print(i)
#         print(raw[i[2],i[0],i[1]])


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


def read_cfar_det(directory):
    with open(directory) as f:
        lines = f.readlines()
        content = [x.split() for x in lines]
    # content = np.asarray(content, dtype=np.float)

    return content

def main():
    data = read_cfar_det('./results/CDC-20200209-124411/2019_09_18_onrd009/rod_res.txt')
    obj_info_list = read_ra_labels_csv('/mnt/nas_crdataset/2019_09_18/2019_09_18_onrd009')
    root_dir_image = './Visualize/'
    file = '2019_09_18_onrd009'
    image_folder = root_dir_image + file + '/'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    # print(obj_info_list)
    start_id = int(data[0][0])
    end_id = int(data[-1][0])
    range_grid = confmap2ra(radar_configs, 'range')
    angle_grid = confmap2ra(radar_configs, 'angle')

    data_list = []

    for i in range(start_id, end_id+1):
        cur_list = []
        for elem in data:
            id = int(elem[0])
            rng = int(elem[2])
            agl = int(elem[3])
            if elem[1] == 'car':
                clas = 2
            elif elem[1] == 'pedestrian':
                clas = 0
            elif elem[1] == 'cyclist':
                clas = 1
            else:
                print('error!!!!!!!!!!!!!!')

            if id == i:
                cur_list.append([rng, agl, clas])

        data_list.append(cur_list)

    # data_list2 = []
    # data2 = read_cfar_det('./results/2019_09_18_onrd004/rod_res.txt')
    # for i in range(start_id, end_id + 1):
    #     cur_list2 = []
    #     for elem in data2:
    #         id = int(elem[0])
    #         rng = int(elem[2])
    #         agl = int(elem[3])
    #         if elem[1] == 'car':
    #             clas = 2
    #         elif elem[1] == 'pedestrian':
    #             clas = 0
    #         elif elem[1] == 'cyclist':
    #             clas = 1
    #         else:
    #             print('error!!!!!!!!!!!!!!')
    #
    #         if id == i:
    #             cur_list2.append([rng, agl, clas])
    #
    #     data_list2.append(cur_list2)

    # plot
    for i in range(start_id, end_id+1):
        print('frame ', i)
        print(data_list[i-start_id])
        print(obj_info_list[i])
        plt.figure()
        # plt.xlim(-90, 90)
        # plt.ylim(1, 27)
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        plt.ylabel('Range /m')
        plt.xlabel('Azimuth angle /degree')
        save_name = image_folder + '/' + file + '_' + str(i).zfill(6)
        for fsi in data_list[i-start_id]:
            c_agl = int(fsi[1])
            c_rng = int(fsi[0])

            c_agl_degree = angle_grid[c_agl]
            c_rng_meter = range_grid[c_rng]
            color = 'b'
            # plt.scatter(c_agl_degree, c_rng_meter, c=color)
            plt.scatter(c_agl, c_rng, c=color)

        # for fsi in data_list2[i-start_id]:
        #     c_agl = int(fsi[1])
        #     c_rng = int(fsi[0])
        #
        #     c_agl_degree = angle_grid[c_agl]
        #     c_rng_meter = range_grid[c_rng]
        #     color = 'g'
        #     # plt.scatter(c_agl_degree, c_rng_meter, c=color)
        #     plt.scatter(c_agl, c_rng, c=color)

        for fsi in obj_info_list[i]:
            c_agl = int(fsi[1])
            c_rng = int(fsi[0])

            c_agl_degree = angle_grid[c_agl]
            c_rng_meter = range_grid[c_rng]
            color = 'r'
            # plt.scatter(c_agl_degree, c_rng_meter, c=color)
            plt.scatter(c_agl, c_rng, c=color)

        # plt.show()
        # input()
        plt.savefig(save_name)
        plt.close()

    # print(data_list)




if __name__ == '__main__':
    n_class = 3
    win_size = 16
    train_model_path = './results'
    # model_dir = 'CDCsc-20200313-104607'
    # model_name = 'rodnet_39_0000037839_000001.pkl'
    model_dir = 'CDC-20200313-104532'
    model_name = 'rodnet_26_0000027013_000001.pkl'
    # model_dir = 'HG-20200316-232953'
    # model_name = 'rodnet_31_0000020787_000001.pkl'
    rodnet = RODNet(n_class=n_class, win_size=win_size).cuda()
    # optimizer = optim.Adam(rodnet.parameters(), lr=lr)
    checkpoint = torch.load(os.path.join(train_model_path, '%s/%s' % (model_dir, model_name)))
    if 'optimizer_state_dict' in checkpoint:
        rodnet.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch_start = checkpoint['epoch']
        # iter_start = checkpoint['iter']
        # loss_cp = checkpoint['loss']
        # if 'iter_count' in checkpoint:
        #     iter_count = checkpoint['iter_count']
    else:
        rodnet.load_state_dict(checkpoint)

    for name, param in rodnet.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
            if name == 'c3d_encode.conv1a.weight':
            # if name == 'stacked_hourglass.conv1a.weight':
                print(param.data[0,:,0,0,0])

    # x = torch.randn(5, 1, 16, 128, 128)
    # y = x.permute(0,2,1,3,4)
    # y = y.view(-1, 1, 128, 128)
    # k = y.view(-1,16,1,128,128)
    # k = k.permute(0,2,1,3,4)
    # print(k.shape)
    # print(y.shape)
    # for i in range(5):
    #     if i == 0:
    #         z = x[i,:,:,:,:].permute(1,0,2,3)
    #     else:
    #         z = torch.cat((z, x[i,:,:,:,:].permute(1,0,2,3)), 0)
    # print(z.shape)
    # print(torch.equal(z,y))
    # print(torch.equal(x, k))
    # input()
    # # with open('/home/admin-cmmb/Documents/RODNet_dop/data/data_details/train/2019_04_09_bms1000.pkl', 'rb') as f:
    # #     data = pickle.load(f)
    # #     print(data[1])
    # #     print(data[0][0] % 3)
    # # input()
    # main()