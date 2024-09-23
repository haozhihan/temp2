torchrun --nproc_per_node=1 --master_port=29500 test_generate.py  \
        --model_name_or_path ../models/gpt2_xl \
        --bf16 True \
        --output_dir ../longlora/tmp/save/s8_0  \
        --dataset_path ../datasets/pg19  \
        --cache_dir ./tmp/cache  \
        --model_max_length 16384 \
        --use_flash_attn True \
        --use_block True \
        --use_shift_head False \
        --use_random_head True \
        --use_random_type 0 \
        --use_random_number 8 \
        --low_rank_training True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 2     \
        --evaluation_strategy "steps"     \
        --save_strategy "steps"     \
        --save_steps 500     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --tf32 True \
        --max_steps 500 \
        --do_eval True  \
        --use_record_attn True  \
        --eval_on_start True  \
        --eval_steps 50 \
        --cross_val True  \
        --deepspeed "ds_configs/stage2.json" 
