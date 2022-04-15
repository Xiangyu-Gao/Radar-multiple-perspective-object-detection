# Radar_multiple_perspective_object_detection
This is a repository for codes and template data of paper ["***Experiments with mmWave Automotive Radar Test-bed***"](https://arxiv.org/pdf/1912.12566.pdf)

***NEW!!! The micro-Dooler classification part has been updated***

Please cite our paper with below bibtex when you find the codes useful.
```
@INPROCEEDINGS{9048939,  author={Gao, Xiangyu and Xing, Guanbin and Roy, Sumit and Liu, Hui}, 
booktitle={2019 53rd Asilomar Conference on Signals, Systems, and Computers}, 
title={Experiments with mmWave Automotive Radar Test-bed}, 
year={2019},  volume={},  number={},  pages={1-6},  doi={10.1109/IEEECONF44664.2019.9048939}}
```

## Software requirement
MATLAB, Python 3.6, Tensorflow 2.0, Jupyter Notebook

## Run codes for generating range-angle maps, range-Doppler maps, and 3D point clouds
1. Customize your testbed/FMCW parameter in script: 
    ```
    ./config/get_params_value.m
    ```
3. Select the input data ('pms1000_30fs.mat', 'bms1000_30fs.mat' or 'cms1000_30fs.mat') in script:
    ```
    generate_ra_3dfft.m
    ```
