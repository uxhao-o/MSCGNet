#!/usr/bin/env bash
source D:/miniconda3/etc/profile.d/conda.sh
conda activate cv
python ../train.py fit \
--trainer.fast_dev_run 100 \
--trainer.log_every_n_steps 14 \
--trainer.max_epochs 10 \
--trainer.callbacks [checkpoint_callback, lr_monitor] \
--trainer.devices 1 \
--trainer.accelerator "gpu" \
--model AgricultureSegmentClass \
--model.model_name "MSCG_Rx50" \
--model.num_classes 7 \
--data AgricultureDataClass \
--data.dataset_root "D:/Agriculture-Vision" \
--data.dataset_name "Agriculture" \
--data.batch_size 10 \
--data.num_workers 4 \
#--ckpt_path "D:\JetBrains\PycharmProjects\CVProjects\cv_02_DR\output\models"