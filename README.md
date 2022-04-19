# Radar_multiple_perspective_object_detection

Automotive Radar Object Recognition by Processing of the Range-Velocity-Angle (RVA) heatmap sequences.


This is a repository for codes and template data of paper ["**RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition**"](https://arxiv.org/pdf/2011.08981.pdf)

Please cite our paper with below bibtex if you find!
 provided codes useful.
```
@ARTICLE{9249018,  author={Gao, Xiangyu and Xing, Guanbin and Roy, Sumit and Liu, Hui},  
journal={IEEE Sensors Journal},   
title={RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition},   
year={2021},  volume={21},  number={4},  pages={5119-5132},  doi={10.1109/JSEN.2020.3036047}}
```

Incomplete. continue to update

## Software requirement
Python 3.6, pytorch-1.5.1, Jupyter Notebook

## 3D slice
for 3D FFT please refer to repo https://github.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification

3. Customize your testbed/FMCW parameter in script: 
    ```
    python slice3d.py
    ```
## Radar Data Augmentation
## Train and Test
1. Download sample data and model from the Google Drive with below link:
    ```
    https://drive.google.com/drive/folders/1TGW6BHi5EZsSCtTsJuwYIQVaIWjl8CLY?usp=sharing
    ```
   Note that we select part of our training and testing set for your use here and the provided model was trainied with whole complete training set. You may use the   above slicing algorithm with 3DFFT data to create your own training and testing set.

2. Decompress the downloaded files and Put the decompressed sample data and trained model in folder as follow:
    ```
    './template_files/data'
    './results/C3D-20200904-001923'
    ```

3. Prepare the input data (RA, RV, and VA slices) and ground truth confidence map for training and testing:
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
