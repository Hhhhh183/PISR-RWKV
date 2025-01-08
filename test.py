import os 
os.environ['CUDA_VISIBLE_DEVICES']='0' 
from evaluation.evaluation_metric import compute_measure
from data_utils import Predata
from model.light_RWKV import Light_RWKV
import numpy as np
import torch
import imageio


# data_root = "/home/user1/LJ/PIns/test"
data_root = "/home/user1/LJ/SR-RWKV/DATASET/B100/LR"  # Dataset Path            
save_dir =  "/home/user1/LJ/SR-RWKV/experiment/weight"  # weighting path
out_root = "/experiment/test_result" # Output Image Path

Generator = Light_RWKV()
Generator.cuda() 


Generator.load_state_dict(torch.load(os.path.join(save_dir, "Weight","Generator_best.pth")),False)  
Generator.eval() 

img_names = os.listdir(data_root)
data_len = len(img_names)
for img_name in img_names:
    lr_img = Predata.load_png(os.path.join(data_root,img_name))

    
    lr_img = ((torch.from_numpy(lr_img)).to(torch.float32)).cuda()
    lr_img = lr_img.unsqueeze(0)


    with torch.no_grad():
        out_img = Generator(lr_img) 

    out_img = out_img.clamp(0, 255) 

    out_img = (out_img.squeeze(0).cpu().numpy()).astype(np.uint8)

   
    out_img = np.transpose(out_img,(1,2,0)) 
    imageio.imwrite(os.path.join(out_root,img_name),out_img)










