## Aspect Ratio and Spatial Layout Aware Image Aesthetics Assessment Using Graph Attention Network
### Overview
There are two stages in the pipeline. 
- **Feature Graph Construction**
![Alt text](figures/Architecture_(a).png "Title")
- **Score Regression using GNN**
![Alt text](figures/Architecture_(b).png "Title")

### Functionalities:
- Extract feature-graphs from AVA images using Inception-Resnet-V2 and 8 augmentations (4 corner crops + flip)
- Train the 6 baseline networks as mentioned in Table 2 of the paper.
- Plot PLCC, SRCC, Accuracy, Balanced Accuracy, MSE Loss, Confusion Matrix, Mean Opinion Score.
- Given an input image/ a set of images and a trained model ([pre-trained weights](https://drive.google.com/drive/folders/10kxMM7etoszz4LxLGczdmqsZORSy1OyV?usp=sharing)), predict the score.

### Environment
- Ubuntu 18.04
- Python 3.6.9
- CUDA 10.2
- pip 20.2.4
- nvidia driver v-450.66

### Installing Dependencies
``sh install_requirements.sh``

The full list of library versions used is provided additionally in the file **requirements.txt**. In case of errors while installing the geometric modules, make sure the pip version matches the one above. The installation of the geometric modules may be slow.

More information is available in the official page:
https://github.com/rusty1s/pytorch_geometric  

### Aesthetic Visual Analysis (AVA) Dataset
There are several crawlers available online for the dataset. For example [here](https://github.com/mtobeiyf/ava_downloader). There are roughly 230K images for training and 20K for testing. We use the same test set as Hosu *et al* [1]. Check the ``meta/`` directory.
### Scripts
#### Feature Graph Construction
`sh extract_graph.sh`

Before running this, specify the necessary parameters inside the script, as commented. A full list of the parameters can be found in ``opts_extractor.py``.
The size of the feature file could go up to 1.5 TB for the entire AVA dataset with 8 augmentations per image.
#### Score Regression using GNN
Once the features are extracted, a GNN can be trained using 

`sh train.sh`

Specify the necessary parameters in the script, as commented. A full list of training parameters can be found in
``opts_train.py``.
We provide all the baseline models in Table 2 of the paper in ``Graph_Models.py``. A 
particular model can be chosen by setting the `MODEL` variable in `train.sh`.

During training, the model tests itself on the AVA test test after ``VAL_AFTER`` epochs. 

#### Visualization
The results can be monitored during training by opening tensorboard separately.

`tensorboard --logdir path/to/visuals/`

If everything works correctly, it should look like this


| Plots  | Images |
| ------------- | ------------- |
| ![Alt text](figures/Screenshot%20from%202020-11-21%2016-50-07.png "Title")  | ![Alt text](figures/Screenshot%20from%202020-11-21%2016-58-04.png "Title")  |





### Predict score on a single or set of images using a trained model
``sh predict_images.sh``

The images are to be copied in the ``DIR`` directory, as specified in ``predict_images.sh``. For example, ``samples`` as in the script.

Pre-trained weights are available [here](https://drive.google.com/drive/folders/10kxMM7etoszz4LxLGczdmqsZORSy1OyV?usp=sharing).
### References
1. Hosu, V., Goldlucke, B. and Saupe, D., 2019. Effective aesthetics prediction with multi-level spatially pooled features. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9375-9383).