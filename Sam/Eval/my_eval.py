import os
import sys
from functools import partial

path_to_check = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if path_to_check not in sys.path:
    sys.path.append(path_to_check)
    
import evaluate
import numpy as np
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from transformers import DataCollatorForLanguageModeling, Trainer

from Sam.models.gpt2_profile_attn_shift import GPT2LMHeadModel
from Sam.utils.args_utils import ModelArguments, TrainingArguments
from Sam.utils.config_utils import get_gpt2_profile_attn_shift_config
from Sam.utils.others import (get_shift_heads_idx, load_jsonl,
                              smart_tokenizer_and_embedding_resize,
                              tokenize_fn)
from Sam.utils.profile_utils import trace_gpt_attn
from Sam.utils.trainer_utils import BothEvalTrainer
import math
from peft import PeftModel

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"   #默认填充
DEFAULT_EOS_TOKEN = "</s>"    #句子结束
DEFAULT_BOS_TOKEN = "<s>"       #句子开始
DEFAULT_UNK_TOKEN = "<unk>"     #未知
def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()


    model_name = model_args.model_type    
    init_checkpoint = '../models/' + model_name + '/pytorch_model.bin'

    config = get_gpt2_profile_attn_shift_config(model_name=model_args.model_name_or_path,
                                                flash_attention=training_args.flash_attention,
                                                use_block=training_args.use_block,
                                                block_ratio=training_args.block_ratio,
                                                use_shift=training_args.use_shift,
                                                use_profile=training_args.use_profile,
                                                max_position_embeddings=training_args.model_max_length,)
    config.move_heads, config.keep_heads = get_shift_heads_idx(total_heads=config.n_head,
                                                              shift_type=training_args.shift_type,
                                                              shift_number=training_args.shift_number)
    
    print("config is !!!!!!")
    print(config)
    model = GPT2LMHeadModel(config)
    
    #load weight
    state_dict = torch.load(init_checkpoint, map_location='cpu')

    for k, v in state_dict.items():
        k = "transformer." + k
        if k in model.state_dict():
            # print('load', k)
            if 'wpe' in k:
                new_length = training_args.model_max_length
                new_wpe_weight = torch.nn.functional.interpolate(v.permute(1, 0).unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0).permute(1, 0)
                v=new_wpe_weight
            model.state_dict()[k].copy_(v)
        else:
            print('skip', k)
    # model.to('cuda')
    print('model name:', model_name)
    
    model = PeftModel.from_pretrained(
        model,
        "experiment/best_forward/output/yes4/checkpoint-500",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    from safetensors.torch import load_file 
    weight = load_file("experiment/best_forward/output/yes4/checkpoint-500/wpe.pt")
    print(model.state_dict().keys())
    for k,v in weight.items():
        # print(k)
        k = "base_model.model."+ k
        if k in model.state_dict():
            model.state_dict()[k].copy_(v)
        else:
            print('skip2', k)
            
    print(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    #增加特殊符号
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    rank = int(os.environ.get('RANK', -1))# 当前进程的排名
    if rank > 0:
        barrier()
    # dataset = load_dataset("/home/pairshoe/cxy/datasets/wiki/wikitext-2-v1")
    dataset_dict = {}
    for split_name in ["train", "validation", "test"]:
        file_path = os.path.join('./data/pg19_2048', f"{split_name}.jsonl")
        if os.path.exists(file_path):
            data = load_jsonl(file_path)
            if data:  # 如果数据集不为空
                dataset_dict[split_name] = Dataset.from_dict({k: [d[k] for d in data] for k in data[0]})

    # 创建 DatasetDict
    dataset = DatasetDict(dataset_dict)
    
    if model_args.cross_val:
        valdata  = load_dataset("../datasets/proof-pile")
        valdata = valdata.map(partial(tokenize_fn,tokenizer),batched=False, num_proc=32, remove_columns=["text"])
    # dataset = load_from_disk("/home/pairshoe/cxy/save/cache/data")
    # dataset = dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=16,remove_columns=["text"])
    if rank == 0:
        barrier()

    print(dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # NOTE prepare to eval
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    metric = evaluate.load("./Sam/Eval/QA/accuracy")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)
    
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    trainer = BothEvalTrainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset1=dataset['validation'],
        eval_dataset2=valdata['test'].select(range(200)),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator)

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=trainer.eval_dataset1)

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

        metrics = trainer.evaluate(eval_dataset=trainer.eval_dataset2)

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
        
        
if __name__ == "__main__":
    train()
