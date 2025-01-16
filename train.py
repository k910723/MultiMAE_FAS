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
results_path = '/shared/shared/yitinglin/PR/' + results_filename

os.system("rm -r "+results_path)

lr_rate1 = random.choice([7e-5,8e-5])#,0.9,
lr1 = lr_rate1

# batch_size = random.choice([10])
batch_size = 32
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
logging.info(f"Batch Size:      {batch_size}")
logging.info(f"Train on {dataset1}")

if dataset1 == 'C' or dataset1 == 'W' or dataset1 == 'S':
    root = '/shared/shared/domain-generalization-multi'
    
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

live_loader = get_loader(root = root, protocol=[dataset1], batch_size=int(batch_size*0.5), shuffle=True, size = 224, cls = 'real')
spoof_loader = get_loader(root = root, protocol=[dataset1], batch_size=int(batch_size*0.5), shuffle=True, size = 224, cls = 'spoof')

# iternum = len(live_loader)
iternum = max(len(live_loader), len(spoof_loader))

live_loader = get_inf_iterator(live_loader)
spoof_loader = get_inf_iterator(spoof_loader)

save_index = 0
log_step = 10
logging.info(f"iternum={iternum}")

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

        pred = multimae(input_dict) # multimae forward
        pred = pred[0]['cls'] # class token

        Crossentropy = criterionCls(pred, labels_rand)
        TotalLossALL = Crossentropy
        
        optimizerALL.zero_grad()
        TotalLossALL.backward()
        optimizerALL.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  Crossentropy: %.4f  TotalLossALL: %.4f'
                         % (epoch + 1, step + 1, Crossentropy.item(), TotalLossALL.item()))      
        
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
            