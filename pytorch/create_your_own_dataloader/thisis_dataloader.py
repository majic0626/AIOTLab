import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt # just for visualize

class OurDataset(Dataset):
    
    def __init__(self, root_path, transforms=None):
        self.root_path = root_path
        self.total_datas, self.class_index = self.get_datas_info(root_path)
        self.transforms = transforms # data preprocessing
        
    def __len__(self):
        return len(self.total_datas)
        
    def __getitem__(self, idx):
        file_path = self.total_datas[idx]['path']
        label = self.total_datas[idx]['label']
        img_arr = Image.open(file_path) # read image
        print(img_arr)
        if self.transforms is not None:
            for t in self.transforms:
                img_arr = t(img_arr)
                
        return img_arr, label
            
    def get_datas_info(self, root_path):
        classes  = os.listdir(root_path)
        class_index = {} # to record class(str) and label(int) relationship
        total_datas = []
        for label, c in enumerate(classes):
            if c[0] == ".":
                continue
            files = os.listdir(root_path+"/"+c+"/")
            for file in files:
                this_data = {"path":root_path+"/"+c+"/"+file, "label":label}
                total_datas.append(this_data)
                
            if str(label) not in class_index:
                class_index[str(label)] = c
        
        return total_datas, class_index
   

if __name__ == "__main__": 
	
    our_dataset = OurDataset("./dataset/", transforms=[transforms.ToTensor()])    

    loader_params = {"batch_size":1,
                     "shuffle":True,
                     "num_workers":1}

    dataloader = DataLoader(our_dataset, **loader_params)

    for datas, labels in dataloader:
        print(datas.size(), labels.size())