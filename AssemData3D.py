import os
from typing import TypeVar, Sequence, List

import torch
from torch.utils.data import Dataset, Subset
from torch._utils import _accumulate
from torchvision import transforms

import cv2
from rgblabel import bgr2class
import patch.edgedetection as edgedetection


T = TypeVar('T')

class AssemDataset(Dataset): 
    def __init__(self, imageDir, labelDir, transform=None):
        self.image_dir = imageDir
        self.label_dir = labelDir
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_namelist = os.listdir(self.image_dir)
        img_namelist.sort(key=lambda arr:(int(arr[0]), int(arr[2]), arr[3], int(arr[:-4])))
        label_namelist = os.listdir(self.label_dir)
        label_namelist.sort(key=lambda arr:(int(arr[0]), int(arr[2]), arr[3], int(arr[:-4]))) 
        img_name = img_namelist[idx]
        label_name = label_namelist[idx]
        
        imgA = cv2.imread(self.image_dir+'/'+img_name)   # 0表示以灰度模式加载图片，默认值1表示彩色模式
        #imgA = cv2.resize(imgA, (160, 160))
        if self.transform:
            imgA = self.transform(imgA)
        imgA_gray = cv2.imread(self.image_dir+'/'+img_name, 0)
        imgA_can = edgedetection.canny(imgA_gray)        # Canny edge detection    
        
        imgB = cv2.imread(self.label_dir+'/'+label_name)
        #imgB = cv2.resize(imgB, (160, 160))
        imgB = bgr2class(imgB)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)  
        
        return imgA, imgA_can, imgB

# Split the dataset in a non-random way, in comparison with torch.utils.data.random_split
def fixed_split(dataset: Dataset[T], lengths: Sequence[int]) -> List[Subset[T]]:
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = range(sum(lengths))
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Generate the 3 datasets
assem_1 = AssemDataset('datadir/img_384_384_1', 'datadir/lbl_384_384_1', transform)
assem_2 = AssemDataset('datadir/img_384_384_2', 'datadir/lbl_384_384_2', transform)

train_size_1 = 1804
val_size_1 = 600
test_size_1 = 600
train_size_2 = 1768
val_size_2 = 590
test_size_2 = 646

train_dataset_1, val_dataset_1, test_dataset_1 = fixed_split(assem_1, [train_size_1, val_size_1, test_size_1])
train_dataset_2, val_dataset_2, test_dataset_2 = fixed_split(assem_2, [train_size_2, val_size_2, test_size_2])


if __name__ =='__main__':
    pass
