# bash experiment/best_forward/scripts/run_record.sh > ./tmp/1.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/2.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/3.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/4.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/5.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/6.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/7.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/8.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/9.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/10.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/11.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/12.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/13.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/14.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/15.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/16.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/17.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/18.log
# bash experiment/best_forward/scripts/run_record.sh > ./tmp/19.log
#!/bin/bash

# 定义种子列表
seeds=(0 1 2 3 7 9 13 17 19 23 29 31 37 42 51 57 63 71 73 79 83 89 97 101 111 123 127 137 149 151)

# 循环遍历每个种子并执行命令
for seed in "${seeds[@]}"
do
    echo "Running experiment with seed $seed"
    python experiment/best_forward/gpt2_best_forward_record.py \
        --model_name_or_path ../models/gpt2_xl \
        --bf16 True \
        --output_dir ./experiment/best_forward/output/yes4 \
        --dataset_path ../datasets/pg19 \
        --cache_dir ./tmp/cache \
        --model_max_length 2048 \
        --flash_attention False \
        --block_number 16 \
        --low_rank_training True \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 300 \
        --save_total_limit 100 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --lr_scheduler_type constant_with_warmup \
        --logging_steps 1 \
        --tf32 True \
        --max_steps 800 \
        --do_eval True \
        --cross_val True \
        --the_seed_is $seed > "./tmp/seed${seed}.log" 

    echo "Experiment with seed $seed finished. Log saved to seed${seed}.log"
done
