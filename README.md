[![Made withPython](https://img.shields.io/badge/Made%20with-python-407eaf?style=for-the-badge&logo=python)](https://www.python.org/)
[![Made withPytorch](https://img.shields.io/badge/Made%20with-pytorch-ee4c2c?style=for-the-badge&logo=pytorch)](https://www.pytorch.org/)
[![Made withCuda](https://img.shields.io/badge/Made%20with-cuda-76b900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)
[![Made withColab](https://img.shields.io/badge/Made%20with-Colab-ee4c2c?style=for-the-badge&logo=Colab)](https://colab.research.google.com/)
[![Made withPython](https://img.shields.io/badge/Made%20with-PaperSpace-407eaf?style=for-the-badge&logo=pytho)](https://www.paperspace.com/)




# Enhanced Pseudo Li-DAR for 3D Image-Based 3D Object Detection
Our work uses stereo images to detect 3D objects passing through 3 main stages:

1-Depth estimation using [DeepPruner](https://github.com/uber-research/DeepPruner) model

2-Changing the 2D depth map to 3D point cloud Lidar representation

3- 3D object detection on the generated point cloud using [CIA-SSD](https://github.com/Vegeta2020/CIA-SSD/).


![ezgif com-gif-maker](https://github.com/a-akram-98/E-P-L/blob/master/out.gif?raw=true)

## Results

```
car  AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:92.95, 76.35, 67.44
bev  AP:75.70, 51.80, 44.03
3d   AP:65.50, 46.70, 39.05
aos  AP:88.77, 72.99, 62.85
```

### Contribution

Our contribution is enhancing the inference of the [old](https://github.com/mileyan/pseudo_lidar) model to achieve real-time and higher accuracy (mAP).
The [old](https://github.com/mileyan/pseudo_lidar/blob/master/preprocessing/generate_lidar.py) CoR didn't support voxelization gradients, Our new CoR supports End-To-End (E2E) with differentiable Voxelization Layer.


## Installation

we're providing a [environment.yml](https://github.com/a-akram-98/E-P-L/blob/master/E2E/environment.yml) file  to setup on the environment with conda with following command:

```bash 
conda env create -f environment.yml
```
Then Follow the setup procedure of [CIA-SSD](https://github.com/Vegeta2020/CIA-SSD/) model
please note that the SPConv model used is provided, please note, in the process of setting up CIA-SSD


please install torch_scatter 1.3.1 version with the following command
```bash
pip install torch_scatter==1.3.1
```

```Our setup is P5000 GPU, Ubuntu Linux 20.04```


## Dataset preparation

```
# For KITTI Dataset
└── KITTI_DATASET
       ├── training    <-- 7481 train data
       |   ├── image_2 
       |   ├── image_3
       |   ├── calib
       |   ├── label_2
       |   └── planes
       |   
       └── testing     <-- 7580 test data
           ├── image_2 
           └── calib
```

## Training
### Depth model
The DeepPruner depth model trained from scratch ```100 epoch``` with the same hyperparameters provided in the paper.

[Depth Pretraining weights](https://drive.google.com/file/d/1OoN5S8qAtpmLWoCzoP0GgfPf1orFI7jP/view)

### 3D Object Detection


The CIA-SSD detection trained from scratch ```60 epoch``` due to change of representation doesn't provide reflectance feature that is provided by KITTI Dataset and CIA-SSD model was trained on. So we had to change the training number of features to be 3 features only ```x, y and z coordinates```.

[3D OD pretraining weights](https://drive.google.com/file/d/1o5RKWjl3x9iRTbeu5HbQQOTua9wTtdrB/view?usp=sharing)

### End-To-End training
After pretraining each model separately as we mentioned we train the whole model in E2E mode for better accuracy.

The following weights is the result of the final model we achieved after the End-To-End training.

[Depth Estimation E2E weights](https://drive.google.com/file/d/1fHt0c5sihOgFkAG2vfpBjqKzPGFaN4wv/view?usp=sharing)

[3D OD E2E weights](https://drive.google.com/file/d/1aTkz-xoT33ftctioSxAP---3BHAuLJfS/view?usp=sharing)


### Scripts
```All the paths in the project are absolute paths not relative paths please consider changing them, you can find these paths in (E2E/trainDP.py), (E2E/testDp.py), (E2E/dp_interface.py) , (E2E/cor_interface.py), (E2E/cia_interface.py), and the config file of CIA-SSD found in (cia/examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py)```

#### Train script

```bash
conda activate cia
cd E2E
python trainDp.py --od_cp /path/to/od/weights --depth_cp /path/to/depth/weights
```
The model supports tensorboardX for **Live** loss curves using the following command in another terminal:

```bash
tensorboard dev upload --logdir ./logs
```

```Note: in E2E train mode when  if you paused the training and tried to resume it please change the flag --load_only_checkpoint to False and --Resume_Training to True```

#### Test script

```bash
conda activate cia
cd E2E
python testDp.py --od_cp /path/to/od/weights --depth_cp /path/to/depth/weights
```
## Visualiztion

We used this [repo]( https://github.com/kuixu/kitti_object_vis) provided by [kuixu](https://github.com/kuixu) to visualize results


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
