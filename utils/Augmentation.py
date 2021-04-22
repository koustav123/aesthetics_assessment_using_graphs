import sys
import torch

sys.path.append('../utils');
from torchvision import transforms
import Custom_Transforms


# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class Augmentation:
    
    def __init__(self, strategy, data_precision):
        self.strategy = strategy;
        self.data_precision = data_precision
        # print("Data Augmentation Initialized with strategy %s" % (self.strategy));

    
    def applyTransforms(self):

        if self.strategy == "MLSP_AUG_FULL_IMAGE": # Dimensions are Resized to 224,224
            data_transforms = {
            'train': transforms.Compose([
                Custom_Transforms.MLSP_FULL(self.data_precision, MEAN, STD)
            ]),
            'test': transforms.Compose([
                Custom_Transforms.MLSP_FULL(self.data_precision, MEAN, STD)
            ]),
        }

        elif self.strategy == "MLSP_8_NUMPY":
            data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                Custom_Transforms.SetTensorPrecision(self.data_precision),
                transforms.Lambda(lambda x: x.permute([ 1, 0, 2]))
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                Custom_Transforms.SetTensorPrecision(self.data_precision),
                transforms.Lambda(lambda x: x.permute([ 1, 0, 2]))

            ]),
        }

        elif self.strategy == "NO_TARGET_TRANSFORM":
            data_transforms = {'train': None, 'test': None}
        # ************* Add Custom Transforms here with an elif ************************

        else :
            print ("Please specify correct augmentation strategy MLSP_AUG_FULL_IMAGE");
            exit();
            
        return data_transforms;