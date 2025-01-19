# python train.py --train_dataset train --total_epoch 1
# python test.py --train_dataset train --test_dataset test --missing none

import torch
import torch.optim as optim
import torch.nn as nn
import itertools
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import logging
from pytz import timezone
from datetime import datetime
import sys
import torchvision.transforms as T
from thop import profile
from balanceloader import *
import warnings
warnings.filterwarnings("ignore")
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
from torch.optim.lr_scheduler import StepLR
from pytorch_metric_learning import losses
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import argparse 

from functools import partial
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import LinearOutputAdapter
from multimae.multimae import pretrain_multimae_base

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--total_epoch", type=int, default=5)
args = parser.parse_args()

dataset1 = args.train_dataset
device_id = 'cuda:0'
results_filename = dataset1.replace('/', '_') + '_MultiMAE' #_final_version_
results_path = '/home/kevin/MMA-FAS/' + results_filename

os.system("rm -r "+results_path)

lr_rate1 = random.choice([7e-5,8e-5])#,0.9,
lr1 = lr_rate1

# batch_size = random.choice([10])
batch_size = 8
model_save_step = 10
model_save_epoch = 1

mkdir(results_path)
mkdir('logger')
#file_handler = logging.FileHandler(filename='/home/s113062513/PR/logger/'+ results_filename +'_train.log')
file_handler = logging.FileHandler(filename='logger/'+ results_filename +'_train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
logging.info(f"Batch Size : {batch_size}")
logging.info(f"Train on {dataset1}")

if dataset1 == 'C' or dataset1 == 'W' or dataset1 == 'S':
    root = '/var/mplab_share_data/domain-generalization-multi/'
if dataset1 == 'train':
    root = '/var/mplab_share_data/flexible_multi_modality/'
    
# if dataset 1 is intraS
if dataset1 == 'intraS':
    # print('intraS')
    root = '/shared/shared/SURF_intra2/'

DOMAIN_CONF = {
    'rgb': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
    },
    'depth': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1), # 1 -> 3 channels in dataloader
    },
    'ir': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1, image_size=224)} # 1 -> 3 channels in dataloader
}
DOMAINS = ['rgb', 'depth', 'ir']

input_adapters = {
    domain: dinfo['input_adapter'](
        patch_size_full=16,
    )
    for domain, dinfo in DOMAIN_CONF.items()
}

output_adapters = {
    'cls': LinearOutputAdapter(
        num_classes=2, 
        use_mean_pooling=True,
        init_scale=1.0
    )
}

multimae = pretrain_multimae_base(
    input_adapters=input_adapters,
    output_adapters=output_adapters,
)

ckpt = torch.load("deit_b2multimae.pth", weights_only=True) # Load DeiT-base pretrained weight
multimae.load_state_dict(ckpt['model'], strict=False)
multimae = multimae.to(device_id)
multimae.train()

criterionCls = nn.CrossEntropyLoss().to(device_id)
cosinloss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device_id)
criterion_mse = nn.MSELoss().to(device_id)
contrastiveloss = losses.NTXentLoss(temperature=0.07)
optimizerALL        = optim.AdamW(multimae.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)

live_loader = get_loader(root = root, protocol=[dataset1], batch_size=int(batch_size*0.5), shuffle=True, size = 224, class_type = 'real')
spoof_loader = get_loader(root = root, protocol=[dataset1], batch_size=int(batch_size*0.5), shuffle=True, size = 224, class_type = 'spoof')

# iternum = len(live_loader)
iternum = max(len(live_loader), len(spoof_loader))

live_loader = get_inf_iterator(live_loader)
spoof_loader = get_inf_iterator(spoof_loader)

save_index = 0
log_step = 10
logging.info(f"iternum={iternum}")

def compute_lbp_features(x):
    """
    Compute Local Binary Pattern (LBP) features for batched feature maps
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
    Returns:
        LBP features with same shape as input
    """
    batch_size, channels, height, width = x.shape
    
    # Pad input to handle boundaries
    x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
    
    # Define 8 neighbors' positions in clockwise order starting from top-left
    positions = [
        (-1, -1), (-1, 0), (-1, 1),  # top row
        (0, 1),                       # right
        (1, 1), (1, 0), (1, -1),     # bottom row
        (0, -1)                       # left
    ]
    
    # Get the 8 neighboring pixels for each position
    neighbors = []
    for dy, dx in positions:
        neighbors.append(x_padded[:, :, 1+dy:height+1+dy, 1+dx:width+1+dx])
    
    # Stack neighbors along a new dimension
    neighbors = torch.stack(neighbors, dim=-1)  # (batch, channels, height, width, 8)
    
    # Get center pixels
    center = x.unsqueeze(-1)  # (batch, channels, height, width, 1)
    
    # Compare with center pixel to get binary pattern
    binary = (neighbors >= center).float()  # (batch, channels, height, width, 8)
    
    # Convert binary pattern to decimal using powers of 2
    # Powers arranged in clockwise order: [1,2,4,8,16,32,64,128]
    weights = 2 ** torch.arange(8, device=x.device)
    lbp = torch.sum(binary * weights, dim=-1)  # (batch, channels, height, width)
    
    return lbp

for epoch in range(args.total_epoch):
    for step in range(iternum):

        # ============ one batch extraction ============#
        rgb_img_live, depth_img_live, ir_img_live, labels_live, atktype_live = next(live_loader)
        rgb_img_spoof, depth_img_spoof, ir_img_spoof, labels_spoof, atktype_spoof = next(spoof_loader)
        # ============ one batch extraction ============#
        
        # if step == 0:
        rgb_img = torch.cat([rgb_img_live,rgb_img_spoof], 0).to(device_id)
        depth_img = torch.cat([depth_img_live,depth_img_spoof], 0).to(device_id)
        ir_img = torch.cat([ir_img_live,ir_img_spoof], 0).to(device_id)
        labels = torch.cat([labels_live,labels_spoof], 0).to(device_id)
        atktype = torch.cat([atktype_live,atktype_spoof], 0).to(device_id)
        
        batchidx = list(range(len(rgb_img)))
        random.shuffle(batchidx)

        rgb_img_rand = NormalizeData_torch(rgb_img[batchidx, :])
        depth_img_rand = NormalizeData_torch(depth_img[batchidx, :])
        ir_img_rand = NormalizeData_torch(ir_img[batchidx, :])
        labels_rand = labels[batchidx]
        # atktype_rand = atktype[batchidx]
        
        input_dict = {}
        input_dict['rgb'] = rgb_img_rand
        input_dict['depth'] = depth_img_rand
        input_dict['ir'] = ir_img_rand

        # batch-level masking branch
        modality_to_masked = torch.randint(0, 3, (1,), device=device_id).item()
        mask_batch = torch.full((batch_size,), modality_to_masked, device=device_id)
        pred_batch, encoder_tokens_batch = multimae(input_dict, masking=mask_batch) # multimae forward
        pred_batch = pred_batch['cls'] # class token

        # sample-level masking branch
        mask_sample = torch.randint(0, 3, (batch_size,), device=device_id)
        pred_sample, encoder_tokens_sample = multimae(input_dict, masking=mask_sample) # multimae forward
        pred_sample = pred_sample['cls'] # class token

        # Extract LBP features for spoof samples
        encoder_tokens = torch.cat([encoder_tokens_batch, encoder_tokens_sample], dim=0)
        encoder_tokens = encoder_tokens.transpose(1, 2)
        B, C, N = encoder_tokens.shape
        modality_tokens = {}
        for i, modality in enumerate(['rgb', 'depth', 'ir']):
            start_idx = i * 196
            end_idx = (i + 1) * 196
            modality_tokens[modality] = encoder_tokens[:, :, start_idx:end_idx]
        encoder_tokens_rgb = modality_tokens['rgb'].view(B, C, 14, 14)
        encoder_tokens_depth = modality_tokens['depth'].view(B, C, 14, 14)
        encoder_tokens_ir = modality_tokens['ir'].view(B, C, 14, 14)
        lbp_features_spoof = compute_lbp_features(encoder_tokens_rgb) + compute_lbp_features(encoder_tokens_depth) + compute_lbp_features(encoder_tokens_ir)

        # Prepare positive and negative set for contrastive loss
        anchor = 0
        positive_set = []
        negative_set = []
        labels_rand_double = torch.cat([labels_rand, labels_rand], dim=0)
        mask = torch.cat([mask_batch, mask_sample], dim=0)
        for i in range(batch_size * 2):
            if i >= labels_rand_double.size(0):
                break
            elif i == anchor:
                continue
            elif labels_rand_double[i] == labels_rand_double[anchor] and mask[i] == mask[anchor]:
                positive_set.append(i)
            else:
                negative_set.append(i)

        # Calculate contrastive loss
        # LBP-guided contrastive loss is not implemented
        contrastive_loss = 0
        temperature = 0.07
        numerator = 0
        denominator = 0
        for i in positive_set:
            numerator += torch.exp(F.cosine_similarity(encoder_tokens[anchor, :, -1], encoder_tokens[i, :, -1], dim=0) / temperature)
        for i in negative_set:
            denominator += torch.exp(F.cosine_similarity(encoder_tokens[anchor, :, -1], encoder_tokens[i, :, -1], dim=0) / temperature)
        denominator += numerator
        contrastive_loss = -torch.log(numerator / denominator).mean()

        Crossentropy_batch = criterionCls(pred_batch, labels_rand)
        Crossentropy_sample = criterionCls(pred_sample, labels_rand)
        Crossentropy = (Crossentropy_batch + Crossentropy_sample) / 2

        TotalLossALL = Crossentropy + contrastive_loss * 0.1
        
        optimizerALL.zero_grad()
        TotalLossALL.backward()
        optimizerALL.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d] Crossentropy: %.4f contrastive_loss: %.4f TotalLossALL: %.4f'
                         % (epoch + 1, step + 1, Crossentropy.item(), contrastive_loss.item(), TotalLossALL.item()))      
        
        # Save models per model_save_step
        # if (step + 1) % model_save_step == 0:# and epoch>3:
        #     mkdir(results_path)
        #     save_index += 1
        #     torch.save(multimae.state_dict(), os.path.join(results_path,"MultiMAE-{}.pth".format(save_index)))
            
    # Save models per model_save_epoch
    if (epoch + 1) % model_save_epoch == 0:    
        mkdir(results_path)
        save_index += 1
        torch.save(multimae.state_dict(), os.path.join(results_path,"MultiMAE-{}.pth".format(save_index)))
            