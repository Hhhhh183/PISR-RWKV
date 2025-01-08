from PIL import Image
import numpy as np
import numpy
import random
import pickle
import pandas as pd
from tools import mkdir
import os
import cv2

class Predata:
    

    def load_png(path):
        img = Image.open(path).convert('RGB')
        img = numpy.array(img) # 先转换为数组   H W C
        img = np.transpose(img, (2, 0, 1))  # 将通道放到第一个维度上,转换为 (C, H, W)
        return img

    def random_crop(imgHR,imgLR, patch_size):

        C,H,W = imgLR.shape
        # patch_h,patch_w = patch_size
    
        # # 确保裁剪尺寸不大于原图像尺寸
        # if patch_h > H or patch_w > W:
        #     raise ValueError("裁剪尺寸应小于原始图像尺寸")
    
        # 随机选择裁剪的起始点
        lr_top = H - patch_size
        lr_left = W - patch_size

        if lr_top > lr_left :
            crop_size = lr_left
        else :
            crop_size=lr_top
        
        lr_top = random.randint(0,crop_size)
        lr_left = random.randint(0,crop_size)        
        hr_left = lr_left*4
        hr_top = lr_top*4
        hr_patch_size=patch_size*4
        lr_patches = imgLR[:,lr_top:lr_top + patch_size, lr_left:lr_left + patch_size]
        hr_patches = imgHR[:,hr_top:hr_top + hr_patch_size, hr_left:hr_left + hr_patch_size]

        return lr_patches ,hr_patches    

    # 裁剪成scale的倍数
    def valid_crop(imgLR,imgHR,scale):

        C,H,W = imgLR.shape

        lr_patch_H = (H//scale)*scale # 取整 //  10//3=3
        lr_patch_W = (W//scale)*scale

        lr_top = random.randint(0,(H-lr_patch_H))
        lr_left = random.randint(0,(W-lr_patch_W))

        hr_left = lr_left*4
        hr_top = lr_top*4
        hr_patch_H = lr_patch_H*4
        hr_patch_W = lr_patch_W*4       

        lr_patches = imgLR[:,lr_top:lr_top + lr_patch_H, lr_left:lr_left + lr_patch_W]
        hr_patches = imgHR[:,hr_top:hr_top + hr_patch_H, hr_left:hr_left + hr_patch_W]                

        return lr_patches ,hr_patches 
    
    def test_crop(imgLR,imgHR,scale):

        C,H,W = imgLR.shape

        lr_patch_H = (H//scale)*scale # 取整 //  10//3=3
        lr_patch_W = (W//scale)*scale

        hr_patch_H = lr_patch_H*4
        hr_patch_W = lr_patch_W*4       

        lr_patches = imgLR[:,0:lr_patch_H, 0: lr_patch_W]
        hr_patches = imgHR[:,0:hr_patch_H, 0:hr_patch_W]                

        return lr_patches ,hr_patches 
       
    def save_bin(data, path, use_int=False): 
    # 将数据保存为二进制文件（.bin）
        mkdir(path, is_file=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_txt( s, path):
    # 将字符串 s 写入到文本文件中
        with open(path,'w') as f:
            f.write(s) 

    def save_csv( data_dict, path):
    # 将字典数据保存为 CSV 文件
        mkdir(path, is_file=True)
        result=pd.DataFrame({ key:pd.Series(value) for key, value in data_dict.items() })
        result.to_csv(path)  

    def save(self, data, path): 
        mkdir(path, is_file=True)
        ex = self.getFileEX(path)
        return self.writer[ex](data, path)
    
    def getFileEX(self, s):
        _, tempfilename = os.path.split(s)
        _, ex = os.path.splitext(tempfilename)
        return ex

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees). 增强：水平翻转或旋转（0、90、180、270 度）
    """
    hflip = hflip and random.random() < 0.5  # 50%概率进行水平翻转
    vflip = rotation and random.random() < 0.5  # 50%概率进行垂直翻转
    rot90 = rotation and random.random() < 0.5  # 50%概率进行旋转90度

    def _augment(img):
        if hflip:  # 水平翻转
            cv2.flip(img, 1, img)
        if vflip:  # 垂直翻转
            cv2.flip(img, 0, img)
        if rot90:  # 旋转90度
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # 水平翻转
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1  # 水平翻转时，调整光流的水平方向
        if vflip:  # 垂直翻转
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1  # 垂直翻转时，调整光流的垂直方向
        if rot90:  # 旋转90度
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]  # 调整光流的通道顺序
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]  # 对所有图像进行增强
    if len(imgs) == 1:
        imgs = imgs[0]  # 如果只有一个图像，去掉列表

    if flows is not None:  # 如果有光流数据，也进行增强
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)  # 返回增强状态
        else:
            return imgs

    # def normalize(img): 
    #     d_min = 0
    #     d_max =  255
    #     img = img.clamp(0, 255)
    #     img = (img - d_min)/(d_max - d_min)
        
    #     return img
    # def denormalize(img): 
    #     d_min = 0
    #     d_max =  255  
    #     img = img.clamp(0, 255)      
    #     img = img*(d_max - d_min) + d_min 
    #     return img






      

    