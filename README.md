# Radar_multiple_perspective_object_detection
This is a repository for codes and template data of paper ["**RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition**"](https://arxiv.org/pdf/2011.08981.pdf)

Please cite our paper with below bibtex when you find the codes useful.
```
@ARTICLE{9249018,  author={Gao, Xiangyu and Xing, Guanbin and Roy, Sumit and Liu, Hui},  journal={IEEE Sensors Journal},   title={RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition},   year={2021},  volume={21},  number={4},  pages={5119-5132},  doi={10.1109/JSEN.2020.3036047}}
```

## Software requirement
Python 3.6, Tensorflow 2.0, Jupyter Notebook

## Run codes for generating range-angle maps, range-Doppler maps, and 3D point clouds
1. Customize your testbed/FMCW parameter in script: 
    ```
    ./config/get_params_value.m
    ```
3. Select the input data ('pms1000_30fs.mat', 'bms1000_30fs.mat' or 'cms1000_30fs.mat') in script:
    ```
    generate_ra_3dfft.m
    ```
