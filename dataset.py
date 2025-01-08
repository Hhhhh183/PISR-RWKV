import torch
from torch.utils.data import Dataset
import os
import numpy 
from data_utils import Predata,augment
import glob
from PIL import Image 
from torchvision.transforms import  ToTensor

    
class Train_Data(Dataset):
    def __init__(self, root_dir, patch_size): 

        self.LR_paths = [] 
        self.HR_paths = [] 

        lr_paths = glob.glob(os.path.join(root_dir,"LR_X4_sub", "*.png")) 

        for lr_path in lr_paths:  
                self.LR_paths.append(lr_path)
                self.HR_paths.append(lr_path.replace("LR_X4_sub", "HR_sub"))

        self.length = len(self.LR_paths) 

        self.patch_size = patch_size



    def __len__(self):
        return self.length 

    def __getitem__(self, idx):

        imgLR = Predata.load_png(self.LR_paths[idx])
        imgHR =Predata.load_png(self.HR_paths[idx])        

        imgLR = torch.from_numpy(imgLR)
        imgHR = torch.from_numpy(imgHR)

        # imgLR = Predata.normalize(imgLR)
        # imgHR = Predata.normalize(imgHR)

        imgLR,imgHR = Predata.random_crop(imgHR =imgHR,imgLR =imgLR,patch_size=self.patch_size)
        imgHR,imgLR = augment([imgHR,imgLR])

        return imgLR, imgHR

# 继承自 torch.utils.data.Dataset，用于加载测试数
class Valid_Data(Dataset): 
    def __init__(self, root_dir): 
   
        self.LR_paths = [] 
        self.HR_paths = [] 
        
        lr_paths = glob.glob(os.path.join(root_dir,"LR", "*.png")) 

        for lr_path in lr_paths:  
                self.LR_paths.append(lr_path)
                self.HR_paths.append(lr_path.replace("LR", "HR"))

        
        self.length = len(self.LR_paths) 
        
        # self.patch_size = patch_size


    def __len__(self):
        return self.length 

    # 用于根据索引 idx 获取一对 LQ 和 HQ 图像
    def __getitem__(self, idx):

        
        imgLR = Predata.load_png(self.LR_paths[idx])
        imgHR =Predata.load_png(self.HR_paths[idx]) 

        # 将numpy转换为张量，并裁剪
        imgLR = torch.from_numpy(imgLR)
        imgHR = torch.from_numpy(imgHR)

        # imgLR = Predata.normalize(imgLR)
        # imgHR = Predata.normalize(imgHR)

        # imgLR,imgHR = Predata.random_crop(imgHR =imgHR,imgLR =imgLR,patch_size=self.patch_size)
        # imgHR,imgLR = Predata.valid_crop(imgHR,imgLR)

        return imgLR, imgHR


class DataSampler:
    # 初始化方法，接受一个 DataLoader 对象，并创建一个迭代器
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self
    # 尝试从迭代器中获取一个批次数据，如果迭代器结束，则重新创建一个迭代器并获取新的批次数据
    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # 如果 DataLoader 中的数据采样完了，重新 shuffle 数据
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        return batch

# dataset = Train_Data() 
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True,drop_last=True) 
# data_sampler = DataSampler(data_loader) 


class Test_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["Set14", "B100", "U100","M109"] , use_num = None, target_folder="validation"): 
        
        self.LQ_paths = [] 
        self.HQ_paths = [] 
        
        
        
        for modality in modality_list: 
            tmp_paths = glob.glob(os.path.join(root_dir, modality, target_folder, "LQ", "*.nii")) 
            
            use_num = len(tmp_paths) if use_num is None else use_num
            
            for num in range(use_num): 
                p = tmp_paths[num]
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))  

        self.length = len(self.LQ_paths) 

    def analyze_path(self, path): 
        path_parts = path.split('/') 
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 
        
        modality = path_parts[-4] 
        return modality, base_name
        

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):

       
        imgLQ = Predata.load_png(self.LQ_paths[idx])
        imgHQ =Predata.load_png(self.HQ_paths[idx]) 
        
        modality, file_name = self.analyze_path(self.LQ_paths[idx]) 
        
        # import pdb 
        # pdb.set_trace()
        
        imgLQ = torch.from_numpy(imgLQ).unsqueeze(0) 
        imgHQ = torch.from_numpy(imgHQ).unsqueeze(0)

        return imgLQ, imgHQ, modality, file_name




