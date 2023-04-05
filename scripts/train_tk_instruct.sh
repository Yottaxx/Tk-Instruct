#!/bin/bash
set -x

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
    # --model_name_or_path google/t5-xl-lm-adapt \

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port --include localhost:1,4 src/run_s2s.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path google/t5-large-lm-adapt \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --output_dir t5-large-adapt-lora-experiment-epoch8/ \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 8 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_strategy steps \
    --save_steps 2000 \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-large-lm-adapt-lora-experiment \
    --max_eval_samples 1000 \
