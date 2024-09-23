from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="../models/gpt2_xl")
    model_type: Optional[str] = field(default="gpt2_xl")
    dataset_path: Optional[str] = field(default="../datasets/pg19")
    cross_val : Optional[bool] = field(default=False)
    the_seed_is : Optional[int] = field(default=1)

#训练参数
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # 模型的最大序列长度
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_block: bool = field(
        default=True,
        metadata={"help": "Whether to shift heads."},
    )
    block_ratio: float = field(
        default=0.25,
        metadata={"help": "The ratio of block size."},
    )
    block_number: int = field(
        default=4,
        metadata={"help": "The ratio of block size."},
    )
    use_shift: bool = field(
        default=True,
        metadata={"help": "Whether to shift heads."},
    )
    shift_type: int = field(
        default=0,
        metadata={"help": "0:前 1:中 2:后 3:交叉  1:random."},
    )
    shift_number: int = field(
        default=6,
        metadata={"help": "how many heads to move."},
    )
    use_profile: bool = field(
        default=True,
        metadata={"help": "Whether to record attention."},
    )
    #是否使用LoRA
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    # 除LoRA权重外的其他可训练参数
    trainable_params: str = field(
        default="wpe,ln",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )