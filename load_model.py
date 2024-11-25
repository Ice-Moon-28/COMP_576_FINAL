from transformers import LlamaForCausalLM
import torch
# 加载预训练的 LLaMA 模型
model_name = "meta-llama/Llama-7B-hf"
model = LlamaForCausalLM.from_pretrained(model_name)

# 替换某一层的参数值
target_layer = model.model.layers[0].self_attn.q_proj  # 替换第 1 层的 query 投影矩阵
with torch.no_grad():
    target_layer.weight[:] = 0  # 将权重替换为全 0
    if target_layer.bias is not None:
        target_layer.bias[:] = 0  # 将偏置替换为全 0

# 验证修改结果
print("Modified q_proj weights:")
print(target_layer.weight)