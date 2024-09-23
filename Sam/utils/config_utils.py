from transformers import GPT2Config


def get_gpt2_config(model_name='openai-community/gpt2-xl'):
    config = GPT2Config.from_pretrained(model_name)
    return config

def get_gpt2_profile_attn_shift_config(model_name='openai-community/gpt2-xl',
                                       flash_attention=True,
                                       use_block=True,
                                       block_ratio=1/4,
                                       use_shift=True,
                                       move_heads=None,
                                       keep_heads=None,
                                       use_profile=False,
                                       profile_max_steps=498,
                                       save_profile=None,
                                       max_position_embeddings = 16384):
    config = GPT2Config.from_pretrained(model_name)
    config.flash_attention = flash_attention
    config.use_block = use_block
    config.block_ratio = block_ratio
    config.use_shift = use_shift
    config.move_heads = move_heads
    config.keep_heads = keep_heads
    config.use_profile = use_profile
    config.profile_max_steps = profile_max_steps
    config.save_profile = save_profile
    # config.max_position_embeddings = max_position_embeddings
    config.n_ctx = max_position_embeddings
    config.n_positions = max_position_embeddings
    return config

def get_gpt2_best_forward_config(model_name='openai-community/gpt2-xl',
                                flash_attention=True,
                                block_ratio=1/4,
                                max_position_embeddings=16384):
    config = GPT2Config.from_pretrained(model_name)
    if flash_attention:
        config._attn_implementation_internal = "flash_attention_2"
    config.flash_attention = flash_attention
    config.block_ratio = block_ratio
    config.n_ctx = max_position_embeddings
    config.n_positions = max_position_embeddings
    return config

def get_gpt2_baseline_config(model_name='openai-community/gpt2-xl',
                            flash_attention=False, 
                            max_position_embeddings=16384):
    config = GPT2Config.from_pretrained(model_name)
    config.n_ctx = max_position_embeddings
    config.n_positions = max_position_embeddings
    config.flash_attention = flash_attention
    return config