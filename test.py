import numpy as np
import os
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import torch

import logging
from pytz import timezone
from datetime import datetime
import torchvision.transforms as transforms
import sys
import torch.nn.functional as F
import torch.nn as nn
import gc
from thop import profile
import time
import glob
import argparse
from intradataloader import *
import pandas as pd
from scipy import interpolate
#from loss.loss import *
from functools import partial
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import LinearOutputAdapter
from multimae.multimae import pretrain_multimae_base

np.random.seed(42)
torch.manual_seed(42)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_tpr_at_fpr(labels, predictions, target_fpr=0.001):
    sorted_indices = np.argsort(predictions)[::-1].astype(int)
    sorted_labels = np.array(labels)[sorted_indices]

    TP = np.cumsum(sorted_labels)
    FP = np.cumsum(1 - sorted_labels)
    FN = np.sum(sorted_labels) - TP
    TN = len(sorted_labels) - np.sum(sorted_labels) - FP

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    target_index = np.where(FPR <= target_fpr)[0]
    if len(target_index) == 0:
        return None  # No FPR value is as low as the target
    tpr_at_target_fpr = TPR[target_index[-1]]

    return tpr_at_target_fpr

def calculate_interpolated_tpr(fpr, tpr, fpr_threshold=0.001):
    interpolated_tpr = np.interp(fpr_threshold, fpr, tpr)
    return interpolated_tpr

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

#receive arguments from command line for testing_dataset
parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--test_dataset", type=str)
parser.add_argument("--missing", type=str, default='None')
args = parser.parse_args()

string = args.test_dataset # "RRR"
missing = args.missing # "RRR"

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device_id = 'cuda:0'
batch_size = 500
dataset1 = args.train_dataset
testing_datasets = [string]

results_filename = dataset1.replace('/', '_') + '_MultiMAE' #_final_version_
results_path = '/home/kevin/MMA-FAS/' + results_filename

if dataset1 == 'C' or dataset1 == 'W' or dataset1 == 'S':
    root = '/var/mplab_share_data/domain-generalization-multi/'

if testing_datasets[0] == 'intraS':
    root = '/shared/shared/SURF_intra2/'
if testing_datasets[0] == 'test':
    root = '/var/mplab_share_data/flexible_multi_modality/'

logger_filename = 'MultiMAE_' + dataset1 + '_to_' + string + '_' + missing
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
mkdir('logger/test/')
file_handler = logging.FileHandler(filename='logger/test/'+ logger_filename +'_test.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

DOMAIN_CONF = {
    'rgb': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
    },
    'depth': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
    },
    'ir': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1, image_size=224)}
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

data_loader = get_loader(root = root, protocol=testing_datasets , batch_size=batch_size, shuffle=False, train=False, size = 224)

logging.info(f"# of testing: {len(data_loader)} on {testing_datasets[0]}")
logging.info(f"path: {results_path}")

record = [1,1,1,1,1,1]

length = int(len(glob.glob(results_path + '/*.pth'))) 

model_save_step = 10
model_save_epoch = 1

#create a log list to store the results
log_list = []

with torch.no_grad():
    
    for step in reversed(range(1, length)):
        Net_path = results_path + "/MultiMAE-" + str(step) + ".pth"
        ckpt = torch.load(Net_path, weights_only=True)
        multimae.load_state_dict(ckpt, strict=False)
        multimae = multimae.to(device_id)
        multimae.eval()

        # Uncomment this line if you want to test every "model_save_step" in a single epoch
        # step = step * model_save_step 
    
        score_list = []
        score_list_live = []
        score_list_spoof = []
        Total_score_list_cs = []
        Total_score_list_all = []
        
        label_list = []
        TP = 0.0000001
        TN = 0.0000001
        FP = 0.0000001
        FN = 0.0000001
        for i, data in enumerate(data_loader):
            rgb_img, depth_img, ir_img, labels = data
            rgb_img = rgb_img.to(device_id)
            depth_img = depth_img.to(device_id)
            ir_img = ir_img.to(device_id)
            
            rgb_img     = NormalizeData_torch((rgb_img))
            depth_img   = NormalizeData_torch((depth_img))
            ir_img      = NormalizeData_torch((ir_img))
            
            input_dict = {}
            input_dict['rgb'] = rgb_img
            input_dict['depth'] = depth_img
            input_dict['ir'] = ir_img

            pred = multimae(input_dict, mask_inputs=False)
            pred = pred[0]['cls']
             
            score = F.softmax(pred, dim=1).cpu().data.numpy()[:, 1]  # multi class
            
            for j in range(rgb_img.size(0)):
                score_list.append(score[j])
                label_list.append(labels[j])

        for i in range(0, len(label_list)):
            Total_score_list_cs.append(score_list[i]) 
            if score_list[i] == None:
                print(score_list[i])
        # if there is nan in Total_score_list_cs, print it out
        fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, Total_score_list_cs)
        threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

        for i in range(len(Total_score_list_cs)):
            score = Total_score_list_cs[i]
            if (score >= threshold_cs and label_list[i] == 1):
                TP += 1
            elif (score < threshold_cs and label_list[i] == 0):
                TN += 1
            elif (score >= threshold_cs and label_list[i] == 0):
                FP += 1
            elif (score < threshold_cs and label_list[i] == 1):
                FN += 1

        APCER = FP / (TN + FP)
        NPCER = FN / (FN + TP)
        
        if record[1]>((APCER + NPCER) / 2):
                record[0]=step
                record[1]=((APCER + NPCER) / 2)
                record[2]=roc_auc_score(label_list, score_list)
                record[3]=APCER
                record[4]=NPCER
                record[5]=calculate_tpr_at_fpr(label_list, NormalizeData(score_list))

        log_list.append([step, np.round(APCER, 4), np.round(NPCER, 4), np.round((APCER + NPCER) / 2, 4)])
        #log list append epoch and acer
        # tpr_fpr0001 = calculate_tpr_at_fpr(label_list, NormalizeData(score_list))
        logging.info('[epoch %d] APCER %.4f BPCER %.4f ACER %.4f  AUC %.4f tpr_fpr0001 %.4f'
                % (step, np.round(APCER, 4), np.round(NPCER, 4), np.round((APCER + NPCER) / 2, 4), np.round(roc_auc_score(label_list, score_list), 4) , calculate_interpolated_tpr(fpr, tpr, fpr_threshold=0.001)))
        # logging.info('[epoch 1 step %d] APCER %.4f BPCER %.4f ACER %.4f  AUC %.4f tpr_fpr0001 %.4f'
        #         % (step, np.round(APCER, 4), np.round(NPCER, 4), np.round((APCER + NPCER) / 2, 4), np.round(roc_auc_score(label_list, score_list), 4) , calculate_interpolated_tpr(fpr, tpr, fpr_threshold=0.001)))

#from the log list select 5 best acer
log_list.sort(key=lambda x: x[3])
print(log_list)
'''
# from the log list test the 5 best acer in if epoch %5 == 0:
with torch.no_grad():
    for i in range(5):
        epochs = log_list[i][0] + 5
        if epochs > length:
            epochs = epochs - 5
            
        for j in range(10):
            epoch = epochs - j
            if epoch %10 == 0:
                continue
            Net_path = results_path + "/FASNet-" + str(epoch) + ".tar"
            loaded_dict = torch.load(Net_path)
            Fas_Net.load_state_dict(loaded_dict, strict = False) # map_location=device_id
            Fas_Net.eval()

            score_list = []
            
            score_list_live = []
            score_list_spoof = []
            Total_score_list_cs = []
            Total_score_list_all = []
            
            label_list = []
            TP = 0.0000001
            TN = 0.0000001
            FP = 0.0000001
            FN = 0.0000001
            for i, data in enumerate(data_loader):
                rgb_img, depth_img, ir_img, labels = data
                rgb_img = rgb_img.to(device_id)
                depth_img = depth_img.to(device_id)
                ir_img = ir_img.to(device_id)
                
                rgb_img     = NormalizeData_torch((rgb_img))
                depth_img   = NormalizeData_torch((depth_img))
                ir_img      = NormalizeData_torch((ir_img))
                
                # for i, img in enumerate(rgb_img):
                #     rgb_img[i] = NormalizeData_torch(rgb_img[i])
                #     ir_img[i] = NormalizeData_torch(ir_img[i])
                #     depth_img[i] = NormalizeData_torch(depth_img[i])
                
                Rpred, _, _, _, _, _\
                = Fas_Net(
                                    rgb_img,#)
                                    ir_img, 
                                    depth_img)
                score = F.softmax(Rpred, dim=1).cpu().data.numpy()[:, 1]  # multi class
                # labels = 1-labels
                for j in range(rgb_img.size(0)):
                    score_list.append(score[j])
                    label_list.append(labels[j])
                
        # score_list = NormalizeData(score_list)

            for i in range(0, len(label_list)):
                Total_score_list_cs.append(score_list[i]) 
                if score_list[i] == None:
                    print(score_list[i])
            fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, Total_score_list_cs)
            threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

            for i in range(len(Total_score_list_cs)):
                score = Total_score_list_cs[i]
                if (score >= threshold_cs and label_list[i] == 1):
                    TP += 1
                elif (score < threshold_cs and label_list[i] == 0):
                    TN += 1
                elif (score >= threshold_cs and label_list[i] == 0):
                    FP += 1
                elif (score < threshold_cs and label_list[i] == 1):
                    FN += 1

            APCER = FP / (TN + FP)
            NPCER = FN / (FN + TP)
            
            if record[1]>((APCER + NPCER) / 2):
                    record[0]=epoch
                    record[1]=((APCER + NPCER) / 2)
                    record[2]=roc_auc_score(label_list, score_list)
                    record[3]=APCER
                    record[4]=NPCER
                    record[5]=calculate_tpr_at_fpr(label_list, NormalizeData(score_list))
            
            # tpr_interpolated = interpolate.interp1d(fpr, tpr, kind="linear")

            # # search 10 index that is closest to 0.001 in fpr
            # indexes = np.argsort((fpr - 0.001))[:5]
            # tpr_interpolated 

            # # try:
            # #     tpr_fpr0001 = tpr_interpolated(0.001)
            # # except ValueError:
            # #     tpr_fpr0001 = 0.0

            #log list append epoch and acer
            
            # tpr_fpr0001 = calculate_tpr_at_fpr(label_list, score_list)
            logging.info('[epoch %d] APCER %.4f BPCER %.4f ACER %.4f  AUC %.4f tpr_fpr0001 %.4f'
                % (epoch, np.round(APCER, 4), np.round(NPCER, 4), np.round((APCER + NPCER) / 2, 4), np.round(roc_auc_score(label_list, score_list), 4) , calculate_interpolated_tpr(fpr, tpr, fpr_threshold=0.001)))
'''
logging.info(f"Modalities BEST Epoch {str(record[0])} ACER {str(record[1])} AUC {str(record[2])} APCER {str(record[3])} BPCER {str(record[4])} tpr_fpr0001 {str(record[5])}")
# logging.info(f"Modalities BEST Epoch 1 Step {str(record[0])} ACER {str(record[1])} AUC {str(record[2])} APCER {str(record[3])} BPCER {str(record[4])} tpr_fpr0001 {str(record[5])}")
