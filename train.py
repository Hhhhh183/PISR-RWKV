import os 
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0'
from model.light_RWKV import Light_RWKV
from dataset import Train_Data, Valid_Data, DataSampler
import torch 
from torch import nn
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tools import mkdir
from data_utils import Predata 
from datetime import datetime
from util_calculate_psnr_ssim import calculate_psnr

total_iteration = 500000
val_iteration = 10000
lr = 2e-4

batch_size = 32
patch_size = 48

eps=1e-8
psnr_max=0

save_dir = "/home/user1/LJ/SR-RWKV/experiment" # Save weights and training information paths


def save_model(G_net_model, save_dir, optimizer_G=None, ex=""):
    save_path=os.path.join(save_dir, "Weight")
    mkdir(save_path) 
    G_save_path = os.path.join(save_path,'Generator{}.pth'.format(ex)) 
    torch.save(G_net_model.cpu().state_dict(), G_save_path)
    G_net_model.cuda()

    if optimizer_G is not None:
        opt_G_save_path = os.path.join(save_path,'Optimizer_G{}.pth'.format(ex))
        torch.save(optimizer_G.state_dict(), opt_G_save_path)


train_dataset = Train_Data(root_dir = "/DIV2K",patch_size=patch_size)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
train_sampler = DataSampler(dataloader)
print("train_data length:", train_dataset.length) 

Valid_dataset = Valid_Data(root_dir = "/Set14")
valid_loader = DataLoader(Valid_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0) # batch_size 必须等于1！
print("valid_data length:", Valid_dataset.length)

'''
Testing Code
'''
Generator = Light_RWKV()
Generator.cuda() 

optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08) 
lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-6)
criterion = nn.L1Loss().cuda()

running_loss = []
eval_metrics={
    "psnr":[],
    "ssim":[], 
    }

loss_list = torch.empty((total_iteration),1)
pbar = tqdm(total=int(total_iteration))
sum_psnr = 0
print("################ Train ################")
for iteration in list(range(1, int(total_iteration)+1)):


    l_G=[]
    in_pic,  label_pic = next(train_sampler) 
    
    in_pic = in_pic.type(torch.FloatTensor).cuda()
    label_pic = label_pic.type(torch.FloatTensor).cuda() 

    #################
    #     train G
    #################
    Generator.train()
    optimizer_G.zero_grad() 

    restored = Generator(in_pic) 
    loss_G = criterion(restored, label_pic) 
    
    loss_G.backward()
    optimizer_G.step()

    torch.cuda.empty_cache() 
    lr_scheduler_G.step()

    txt = "%s:iteration [%d]  train Loss : %f," % (datetime.now(),iteration, loss_G.item())
    mkdir(os.path.join(save_dir,"loss"), is_file=True)

    if iteration % val_iteration == 0: 
        sum_psnr=0
        Generator.eval() 
        for counter,data in enumerate(tqdm(valid_loader)):
            val_lr, val_hr = data 
            val_lr = val_lr.type(torch.FloatTensor).cuda() 
            val_hr = val_hr.type(torch.FloatTensor)

            with torch.no_grad():
                val_img = Generator(val_lr) 
                val_img = val_img.clamp(0, 255)
            
            val_hr = val_hr.cuda()
            
            psnr = calculate_psnr(val_img, val_hr, crop_border=4, test_y_channel=True)


            torch.cuda.empty_cache()
            sum_psnr = psnr +sum_psnr

            mean_psnr=sum_psnr/(counter+1)
 
        eval_metrics['psnr'].append(mean_psnr)

        torch.cuda.empty_cache()
        save_model(G_net_model=Generator, save_dir=save_dir, optimizer_G=None, ex="_iteration_{}".format(iteration))
        if mean_psnr>=psnr_max:

            psnr_max=mean_psnr
            Predata.save_txt(("Best Iteration: {}, PSNR: {}".format(iteration, mean_psnr)),os.path.join(save_dir, "best.txt"))
            save_model(G_net_model=Generator, save_dir=save_dir, optimizer_G = optimizer_G, ex="_best")
            Predata.save_bin(
                {'eval_metrics':eval_metrics},
                os.path.join(save_dir, "evaluationLoss.bin")
                )       
                  
    pbar.set_description("loss_G:{:6}, psnr:{:6}".format(loss_G.item(), eval_metrics['psnr'][-1] if len(eval_metrics['psnr'])>0 else 0)) 
    pbar.update()


    
    




