# Use the Data from UWCR dataset

This page introduces how to convert the data format of [UWCR dataset](https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection) to that required by [RAMP-CNN model](https://github.com/Xiangyu-Gao/Radar-multiple-perspective-object-detection). 

## Download the UWCR dataset
Follow the instruction in the UWCR [README](https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection) and decompress the downloaded zip file.

## Convert the Labels
To convert the annotations, you may use the function [convert_annotations.py](../utils/convert_annotations.py). An example of calling this function for the sequence '2019_04_09_cms1000' is presented in the script [convertFormat](../convertFormat.py).
```
    python convertFormat.py
```
The converted labels will be stored under the folder of "2019_04_09_cms1000" as "ramap_labels.csv".