import os
from typing import Optional, Tuple, Union

import torch


group_size_ratio = 1/4
def get_forward_function(use_block = True,
                         use_shift_head=True,
                         use_random_head=False,
                         use_random_type=0,
                         move_heads=None,
                         keep_heads=None,
                         use_record_attn=False,
                         save_attn_dir = None):
    
    if use_shift_head and use_random_head:
        raise ValueError("use_shift_head and use_random_head cannot be used together.")
     
    def forward_with_flash_block(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        bsz, q_len, _ = hidden_states.size()
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        query_length = query.shape[2]
        tgt_len = key.shape[2]
        
        # NOTE :apply shift or block
        group_size = int(q_len * group_size_ratio)
        if q_len % group_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d." % (q_len, group_size))
        num_group = q_len // group_size
            
        def shift(qkv,num_heads,head_dim):
            # qkv = [bsz, nh, q_len, d]
            qkv = qkv.transpose(1, 2)
            # qkv[:, :, num_heads//2:] = qkv[:, :, num_heads//2:].roll(-group_size//2, dims=1)
            if use_shift_head:
                rolled_qkv = qkv[:, :, num_heads//2:].roll(-group_size//2, dims=1)
                qkv = torch.cat([qkv[:, :, :num_heads//2], rolled_qkv], dim=2)
            if use_random_head:
                move_qkv = qkv[:, :, move_heads, :].roll(-group_size//2, dims=1)
                new_qkv = torch.zeros_like(qkv)
                new_qkv[:, :, move_heads, :] = move_qkv
                new_qkv[:, :, keep_heads, :] = qkv[:, :, keep_heads, :]
                qkv = new_qkv
                
            # -> [bsz * n_group, group_s, nh, d)
            #   -> [bsz * n_group, nh, group_s, d)
            qkv = qkv.reshape(bsz * num_group, group_size, num_heads, head_dim).transpose(1, 2)
            return qkv
        # contiguous is required as self._attn() will attempt to apply .view() on them
        if use_block and self.training:
            query = shift(query, self.num_heads,  self.head_dim)
            key = shift(key, self.num_heads,  self.head_dim)
            value = shift(value, self.num_heads,  self.head_dim)
            
            if attention_mask:
                attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
        # NOTE :shift end
        
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        if use_block and self.training:
            # print('shift')
            query = query.transpose(1, 2).view(bsz * num_group, group_size, self.num_heads, self.head_dim)
            key = key.transpose(1, 2).view(bsz * num_group, group_size, self.num_heads, self.head_dim)
            value = value.transpose(1, 2).view(bsz * num_group, group_size, self.num_heads, self.head_dim)
        else:
            # print('no shift')
            query = query.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
            key = key.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
            value = value.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.attn_dropout.p if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        if query.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.c_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)
        
        if use_record_attn:
            attn_output, softmax_lse, S_dmask = self._flash_attention_forward(
                query, key, value, attention_mask, query_length, dropout=attn_dropout
            )
        else:
            attn_output = self._flash_attention_forward(
                query, key, value, attention_mask, query_length, dropout=attn_dropout
            )
        # record the attn if required
        if use_record_attn and self.training:
            with torch.no_grad():
                if self.record_count < self.record_total:
                    self._record_attn(S_dmask, final_save=False)
                    self.record_count += 1
                elif self.record_count == self.record_total:
                    save_attn = os.path.join(save_attn_dir,'attn')
                    os.makedirs(save_attn, exist_ok=True)
                    self._record_attn(S_dmask, final_save=True, save_dir = save_attn)
                    self.record_count += 1
                        
        # NOTE shift back
        if use_block and self.training:
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            # [bsz, q_len, nh, hd]
            if use_shift_head:
                rolled_attn_output = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
                new_attn_output = torch.cat((attn_output[:, :, :self.num_heads//2], rolled_attn_output), dim=2)
                attn_output = new_attn_output
            if use_random_head:
                move_attn_output = attn_output[:, :, move_heads, :].roll(group_size//2, dims=1)
                new_attn_output = torch.zeros_like(attn_output)
                new_attn_output[:, :, move_heads, :] = move_attn_output
                new_attn_output[:, :, keep_heads, :] = attn_output[:, :, keep_heads, :]
                attn_output = new_attn_output
        # NOTE shift back over
        
        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        attn_output = self.c_proj(attn_weights_reshaped)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights_reshaped,)

        return outputs
    
    return forward_with_flash_block
