# Radar_multiple_perspective_object_detection
This is a repository for codes and template data of paper ["**RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition**"](https://arxiv.org/pdf/2011.08981.pdf)

Please cite our paper with below bibtex if you find provided codes useful.
```
@ARTICLE{9249018,  author={Gao, Xiangyu and Xing, Guanbin and Roy, Sumit and Liu, Hui},  
journal={IEEE Sensors Journal},   
title={RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition},   
year={2021},  volume={21},  number={4},  pages={5119-5132},  doi={10.1109/JSEN.2020.3036047}}
```

## Software requirement
Python 3.6, pytorch-1.5.1, Jupyter Notebook

## Run codes for generating range-angle maps, range-Doppler maps, and 3D point clouds
1. Download sample data and model:
    ```
    https://drive.google.com/drive/folders/1TGW6BHi5EZsSCtTsJuwYIQVaIWjl8CLY?usp=sharing
    ```
3. Customize your testbed/FMCW parameter in script: 
    ```
    ./config/get_params_value.m
    ```
3. Prepare the data and ground truth:
    ```
    python prepare_data.py -m train -dd './data/'
    python prepare_data.py -m test -dd './data/'
    ```
4. Run training:
    ```
    python train_dop.py -m C3D
    ```
    You will get training as below
    ```
    No data augmentation
    Number of sequences to train: 1
    Training files length: 111
    Window size: 16
    Number of epoches: 100
    Batch size: 3
    Number of iterations in each epoch: 37
    Cyclic learning rate
    epoch 1, iter 1: loss: 8441.85839844 | load time: 0.0571 | backward time: 3.1147
    epoch 1, iter 2: loss: 8551.98437500 | load time: 0.0509 | backward time: 2.9038
    epoch 1, iter 3: loss: 8019.63525391 | load time: 0.0531 | backward time: 2.9171
    epoch 1, iter 4: loss: 8376.16015625 | load time: 0.0518 | backward time: 2.9146
    ```
5. Run testing:
    ```
    python test.py -m C3D -md C3D-20200904-001923
    ```
6. Run evaluate:
    ```
    python evaluate.py -md C3D-20200904-001923
    ```
