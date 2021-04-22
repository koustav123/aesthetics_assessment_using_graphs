from torchvision import transforms
import torch

class MLSP_FULL(object):
    def __init__(self, data_precision, MEAN, STD):
        self.data_precision = data_precision
        self.mean = MEAN
        self.std = STD

    def __call__(self, img):
        #pdb.set_trace()
        w, h = img.size
        new_w, new_h = int(.875 * w), int(.875 * h)
        # un-comment next line only for nasnetlarge
        new_w, new_h = new_w - new_w % 16, new_h - new_h % 16


        five_crop = transforms.FiveCrop([new_h, new_w])
        flip = transforms.RandomHorizontalFlip(p=1.0)
        tensorize = transforms.ToTensor()
        normalize = transforms.Normalize(self.mean, self.std)
        precise = SetTensorPrecision(self.data_precision)

        crops = five_crop(img)
        final_crops = list(crops[0:4]) + [flip(c) for c in crops][0:4]
        tensors = precise((torch.stack([normalize(tensorize(i)) for i in final_crops])))
        return tensors

class SetTensorPrecision(object):
    def __init__(self, data_precision):
        self.data_precision = data_precision

    def __call__(self, img_t):
        if self.data_precision == 16:
            img_t = img_t.half()
        else:
            img_t = img_t.float()
        return img_t