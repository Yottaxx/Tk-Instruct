#!/bin/bash
set -x

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
    # --model_name_or_path google/t5-xl-lm-adapt \

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port --include localhost:0,5,6,7 src/run_rl_s2s.py \
    --do_train False \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path google/t5-large-lm-adapt \
    --max_source_length 1024 \
    --max_target_length 256 \
    --logits_shape 256 \
    --generation_max_length 256 \
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
    --output_dir t5-large-lm-adapt-lora-experiment-rl/ \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-large-lm-adapt-lora-experiment-rl \
    --max_rl_sample 1600 \
    --episode 32 \
    --max_eval_samples 3000 \

    # --warmup_steps 0 \
    # --logging_strategy steps \
    # --logging_steps 500 \
    # --evaluation_strategy steps \
    # --eval_steps 800 \
    # --save_strategy steps \
    # --save_steps 800 \
    # --deepspeed ds_configs/stage2.config \
    # --bf16 \