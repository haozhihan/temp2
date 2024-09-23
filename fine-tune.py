# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from llama_attn_replace import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

from forward_gpt2 import forward_gpt

from datasets import load_dataset ,load_from_disk

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"   #默认填充
DEFAULT_EOS_TOKEN = "</s>"    #句子结束
DEFAULT_BOS_TOKEN = "<s>"       #句子开始
DEFAULT_UNK_TOKEN = "<unk>"     #未知

#模型参数
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="gpt2")

#训练参数
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # 模型的最大序列长度
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    #是否使用LoRA
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    # 除LoRA权重外的其他可训练参数
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

#调整tokenizer和嵌入层的大小---目的：加一些特殊符号
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer)) # 调整模型嵌入层的大小以匹配tokenizer的长度

    # 如果有新增的符号，更新这些新符号的嵌入使之与已有的嵌入平均值一致
    if num_new_tokens > 0:
        #已有权重
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        #增加新符号权重
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

#调整分词处理，让其成为长句子
def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=True,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    elif model_args.model_type == "gpt2":
        transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = forward_gpt
    else:
        assert model_args.model_type == "llama", "Only support llama and gpt-neox for now"
        # LLaMA模型的注意力替换
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    print("config is !!!!!!!!!!!!")
    print(config)

    # orig_rope_scaling = getattr(config, "rope_scaling", None) #原始的RoPE缩放配置
    # if orig_rope_scaling is None:
    #     orig_rope_scaling = {"factor": 1}

    # orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    # orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # if orig_ctx_len:
    #     orig_ctx_len *= orig_rope_scaling_factor
    #     if training_args.model_max_length > orig_ctx_len:
    #         scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
    #         config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    orig_ctx_len = getattr(config,"n_ctx",None)
    if training_args.model_max_length > orig_ctx_len:
        config.n_ctx = training_args.model_max_length
        config.n_positions = training_args.model_max_length

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True
    )
    
    if model.transformer.wpe.weight.shape[0] != training_args.model_max_length:
        model.transformer.wpe.weight = torch.nn.Parameter(torch.zeros(training_args.model_max_length, config.n_embd))
        torch.nn.init.normal_(model.transformer.wpe.weight, mean=0.0, std=config.initializer_range)
        
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
    # dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=training_args.cache_dir)
    # dataset = dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=128, remove_columns=["text", "meta"])

    dataset = load_from_disk("/home/pairshoe/cxy/save/cache/data")
    dataset = dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=16,remove_columns=["text"])
    if rank == 0:
        barrier()

    print(dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #定义LoRA目标层
    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        elif model_args.model_type == "gpt2":
            targets = ["c_attn"]
            # targets=[]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params（embed,norm）
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
