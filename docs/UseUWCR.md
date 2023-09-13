# Use the Data from UWCR dataset for RAMP-CNN

This page introduces how to convert the data format of [UWCR dataset](https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection) to that required by [RAMP-CNN model](https://github.com/Xiangyu-Gao/Radar-multiple-perspective-object-detection). 

## Download the UWCR dataset
Follow the instruction in the UWCR [README](https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection) and decompress the downloaded zip file.

## Convert the Labels
To convert the annotations, you may use the function [`convert_annotations.py`](../utils/convert_annotations.py). An example of calling this function for the sequence '2019_04_09_cms1000' is presented in the script [convertFormat](../convertFormat.py).
```
    python convertFormat.py
```
The converted labels will be stored under the folder of "2019_04_09_cms1000" as "ramap_labels.csv".

## Convert the Raw ADC Data
To convert the [ADC data]((https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection)) to the required format for training and testing (i.e.,, similar to the 'train_test_data.zip'), you may need to
1) Convert the ADC data to the format of `samples x antennas x chirps` and save each frame to '.mat' file locally. 

2) Run the [`slice3d.py`](../slice3d.py) to generate RA slice, RV slice, VA slice. *Note that please check if the input data format of  [`slice3d.py`](../slice3d.py) is same to the ADC data in [UWCR dataset](https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection)*

3) Save ave the RA slice, RV slice, VA slice to '.npy' file in the folders `RANPY, RVNPY, VANPY`, respectively, following the below directory structure:
```
train_test_data
--date_1
----seq_1
--------camera image
--------RANPY
--------RVNPY
--------VANPY
--------ramp_labels.csv
----seq_2
    ...
--date_2
----seq_4
    ...

...
```

4) You are done! Go ahead and use the new dataset.