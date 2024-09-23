import json
import random
from typing import Dict

import numpy as np
import torch
import transformers

from Sam.utils.profile_utils import quant_this


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
    return num_new_tokens

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

def get_shift_heads_idx(total_heads=25,
                        shift_type=0,
                        shift_number=6):
    move_heads_ini = list(range(shift_number))
    if shift_type == 0:
        move_heads = move_heads_ini
    elif shift_type == 1:
        move_heads = [i + total_heads//2 - shift_number//2 for i in move_heads_ini]
    elif shift_type == 2:
        move_heads = [i + total_heads - shift_number for i in move_heads_ini]
    elif shift_type == 3:
        step = (total_heads - 1) / (shift_number - 1)
        move_heads = [min(int(round(step * i)), total_heads - 1) for i in range(shift_number)]
        
    keep_heads = [i for i in range(total_heads) if i not in move_heads]
    
    return move_heads, keep_heads


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data
# 定义 pack 和 unpack hook
def pack_hook(tensor):
    dtypee = tensor.dtype
    quantized_data = []
    scales_and_min_vals = []
    float_data = []
    aa = quant_this[0].clone()  # 假设 quant_this 是在作用域中的变量
    flag=0
    device = tensor.device  # 获取输入 tensor 所在的设备

    if tensor.shape[0] != 1:
        tensor = tensor.unsqueeze(0)
        flag=1
    # print(tensor.shape)
    
    # 如果 tensor 的形状是 [1, 25, 2048, 64]，直接返回原 tensor，无需量化
    if tensor.shape == torch.Size([1, 25, 2048, 64]):
        return (tensor, None, None, None,None)
    
    for i in range(16):
        for j in range(16):
            block = tensor[:, :, i*128:(i+1)*128, j*128:(j+1)*128]
            if aa[i, j]:
                if block.numel() > 0:  # 确保块非空
                    # 计算量化的 scale 和 min_val
                    min_val, max_val = block.min(), block.max()
                    scale = (max_val - min_val) / 15  # 15 是 int4 的最大值
                    scale = scale if scale != 0 else 1.0  # 防止除以零
                    
                    # 转换 min_val 和 scale 为张量，并将其移动到正确的设备
                    min_val = torch.tensor(min_val, device=device)
                    scale = torch.tensor(scale, device=device)

                    quantized_block = torch.round((block - min_val) / scale).to(torch.int8)  # 初步量化为 int8

                    # 将两个 int4 值打包到一个 int8 中
                    packed_block = quantized_block.reshape(-1, 2)
                    packed_block = (packed_block[:, 0] & 0x0F) | ((packed_block[:, 1] & 0x0F) << 4)
                    quantized_data.append(packed_block)

                    # 保存量化后的数据和 scale, min_val
                    scales_and_min_vals.append((min_val, scale))
            else:
                if block.numel() > 0:  # 确保块非空
                    # 不需要量化的直接保存
                    scales_and_min_vals.append(0)
                    float_data.append(block)

    # 返回打包后的张量列表
    packed_tensor_int8 = torch.cat(quantized_data).to(device) if quantized_data else torch.tensor([], dtype=torch.int8).to(device)
    packed_tensor_float = torch.cat([b.reshape(-1) for b in float_data]).to(device) if float_data else torch.tensor([], dtype=dtypee).to(device)
    return (packed_tensor_int8, packed_tensor_float, scales_and_min_vals, aa, flag)

def unpack_hook(packed_data):
    packed_tensor_int8, packed_tensor_float, scales_and_min_vals, aa, flag = packed_data
    
    if aa is None:
        # print(f'aa is None, return {packed_tensor_int8.shape}')
        return packed_tensor_int8[0]

    device = packed_tensor_int8.device  # 获取打包张量所在的设备
    unpacked_tensor = torch.zeros(1, 25, 2048, 2048, dtype=packed_tensor_float[0].dtype, device=device)
    offset_int8 = 0
    offset_float = 0

    for i in range(16):
        for j in range(16):
            block_shape = (1, 25, 128, 128)
            block_size = block_shape[0] * block_shape[1] * block_shape[2] * block_shape[3]

            if aa[i, j]:
                if scales_and_min_vals[i * 16 + j] is not None:
                    # 提取量化后的数据并解包
                    packed_block = packed_tensor_int8[offset_int8:offset_int8 + block_size // 2].reshape(-1)
                    quantized_block = torch.empty(block_size, dtype=torch.int8, device=device)
                    quantized_block[::2] = packed_block & 0x0F  # 低 4 位
                    quantized_block[1::2] = (packed_block >> 4) & 0x0F  # 高 4 位
                    quantized_block = quantized_block.reshape(block_shape).to(unpacked_tensor.dtype)

                    min_val, scale = scales_and_min_vals[i * 16 + j]

                    # 解量化
                    unpacked_tensor[:, :, i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = quantized_block * scale + min_val
                    offset_int8 += block_size // 2
            else:
                if block_size > 0:
                    # 提取未量化的数据
                    unpacked_tensor[:, :, i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = packed_tensor_float[offset_float:offset_float + block_size].reshape(*block_shape)
                    offset_float += block_size
    if flag==1:
        # print(unpacked_tensor[0].shape)
        # print('shang')
        return unpacked_tensor[0]
    # print(unpacked_tensor.shape)
    return unpacked_tensor



def select_positions_based_on_sum(vv, target_sum=250):

    # 创建一个布尔类型的 Tensor，用来标记选中的位置
    selected_positions = torch.zeros_like(vv, dtype=torch.bool)

    # 提取左下角（包括对角线）的 136 个元素
    indices = [(i, j) for i in range(16) for j in range(i + 1)]
    values = torch.tensor([vv[i, j] for i, j in indices])

    # 对这些元素从小到大排序
    sorted_indices = torch.argsort(values)
    sorted_values = values[sorted_indices]

    # 累积和到 target_sum 为止
    cumulative_sum = torch.cumsum(sorted_values, dim=0)
    threshold_index = (cumulative_sum <= target_sum).sum().item()

    # 选取前 threshold_index 个元素的位置并标记为 True
    selected_indices = sorted_indices[:threshold_index]
    for idx in selected_indices:
        i, j = indices[idx]
        selected_positions[i, j] = True

    return selected_positions



def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
