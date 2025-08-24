#!/bin/bash
subjects='B112 B305 N304 N310'

for subject in ${subjects}
do
    python Run/Runner_new.py \
        --seed 42 \
        --rm_ch_list 16 17 \
        --class_weight 1 1 \
        --fold_k 1 \
        --batch_size 16 \
        --max_epochs 100 \
        --data_config data/test1/dataConfigStimGRLsubject${subject}.json \
        --comment "GRLsubject_with_CosineAnnealingWarmUpRestarts" \
        --subject_name $subject \
        --gpu 2 \
        --model_explain "20240927 GRL + target classifier + domain classifier. totalloss = target+domain, 100 epoch. class_weight = {0 : 1, 1: 1}" \
        --model_name "EEGNetDomainAdaptation" \
        --monitor_value_name "val_target_loss" \
        --subject_usage "test1" \
        --run_pre_test # 명시적이면 OK
done