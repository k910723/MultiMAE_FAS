#!/bin/bash
#nohup bash /home/kevin/MMA-FAS/MultiMAE_FAS/train_and_test.sh &

python train.py --train_dataset train --total_epoch 5
python test.py --train_dataset train --test_dataset test --missing none