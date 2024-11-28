import functools
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import DataParallel
@functools.lru_cache()
def load_model(model_name, device, cache_dir=None, torch_dtype=torch.float16):
    try:
        if model_name.startswith("meta-llama"):
            cleaned_model_name = model_name.removeprefix("meta-llama/")
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                cache_dir='./weights/' + cleaned_model_name,
                torch_dtype=torch_dtype,
            )
            gpu_count = torch.cuda.device_count()

            if gpu_count > 1:
                model = DataParallel(model, device_ids=[i for i in range(gpu_count)])

                model = model.to(device)

                model.generate = model.module.generate
            else:
                model = model.to(device)

        return model

    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    

@functools.lru_cache()
def load_tokenizer(model_name, cache_dir=None, torch_dtype=torch.float16):
    
    if model_name.startswith("meta-llama"):
        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch_dtype)

    return tokenizer

# def load_model_and_tokenizer(model_name='opt-13b', device='cuda:2', **kwargs):
#     if model_name in {'gpt-3.5-turbo'}:
#         return None, None
#     if model_name in {"opt-2.7b", "opt-1.3b", "opt-6.7b", 'opt-13b'}:
#         return load_model_and_tokenizer(f"facebook/{model_name}", device, **kwargs)
#     if model_name.startswith('facebook/opt-'):
#         return _load_pretrained_model(model_name, device, **kwargs), _load_pretrained_tokenizer(model_name)
#     return _load_pretrained_model(model_name, device, **kwargs), _load_pretrained_tokenizer(model_name)

# def load_tokenizer(model_name='opt-13b', use_fast=False):
#     if model_name in {'gpt-3.5-turbo'}:
#         return None
#     if model_name in {"opt-2.7b", "opt-1.3b", "opt-6.7b", 'opt-13b'}:
#         return load_tokenizer(f"facebook/{model_name}", use_fast=use_fast)
#     if model_name.startswith('facebook/opt-'):
#         return _load_pretrained_tokenizer(model_name, use_fast=use_fast)
#     return _load_pretrained_tokenizer(model_name, use_fast=use_fast)
