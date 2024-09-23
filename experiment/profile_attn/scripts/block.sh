python experiment/profile_attn/train_others.py \
    --model_name_or_path ../models/gpt2_xl \
    --bf16 True \
    --output_dir ./experiment/profile_attn/output/block_2048  \
    --dataset_path ../datasets/pg19 \
    --cache_dir ./tmp/cache \
    --model_max_length 2048 \
    --flash_attention True \
    --use_block True \
    --use_shift False \
    --shift_type 1 \
    --shift_number 6 \
    --low_rank_training True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type constant_with_warmup \
    --logging_steps 1 \
    --tf32 True \
    --max_steps 800 \
    --use_profile False \
    --cross_val True \
    --do_eval True \