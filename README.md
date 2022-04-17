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
Python 3.6, Tensorflow 2.0, Jupyter Notebook

## Run codes for generating range-angle maps, range-Doppler maps, and 3D point clouds
1. Customize your testbed/FMCW parameter in script: 
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
    python train.py -m HG -dd /mnt/ssd2/rodnet/data_refine/ -ld /mnt/ssd2/rodnet/checkpoints/ -sm -md HG-20200122-104604
    ```
