import torch


def knock_out_nth_layer_mth_attention_head(model, layer_index, head_index):
    """
    屏蔽 Transformer 模型第 n 层的第 m 个 attention head。

    参数：
    - model: 预训练的 Transformer 模型（如 Llama, GPT 等）。
    - layer_index: 层的索引，从 0 开始计数。
    - head_index: 需要屏蔽的 attention head 索引，从 0 开始计数。

    返回：
    - 修改后的模型，其中指定层的指定 attention head 被屏蔽。
    """
    # layers in the model

    layers = list(model.named_modules())

    # target layer
    target_layer = None
    for name, module in layers:
        if f"layers.{layer_index}" in name and (hasattr(module, "self_attn") or hasattr(module, "attention")):
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"找不到第 {layer_index} 层的 attention 模块。请检查层索引是否正确。")

    if hasattr(target_layer, "self_attn"):
        attention_layer = target_layer.self_attn
    else:
        attention_layer = target_layer.attention

    # attention head's number and head dimension
    num_heads = attention_layer.num_heads
    if head_index >= num_heads:
        raise ValueError(f"head_index 超出范围：最大值为 {num_heads - 1}，但收到 {head_index}。")
    
    if type(attention_layer).__name__ == 'LlamaSdpaAttention':
        
        head_dim = attention_layer.o_proj.weight.shape[0] // num_heads

        start = head_index * head_dim
        end = (head_index + 1) * head_dim

        with torch.no_grad():
            attention_layer.q_proj.weight[start:end, :] = 0
            attention_layer.k_proj.weight[start:end, :] = 0
            attention_layer.v_proj.weight[start:end, :] = 0

            # 2. 对 o_proj 进行零化
            attention_layer.o_proj.weight[start:end, :] = 0

        print(f"已屏蔽第 {layer_index} 层的第 {head_index} 个 attention head。")
        return model

import matplotlib.pyplot as plt
import torch

def visualize_attention(model, attention_mask, input_ids, m_layers=0, n_heads=0):

    import pdb; pdb.set_trace()
    # 通过模型获得注意力权重
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    attentions = outputs.attentions  # 返回所有层的注意力权重

    m_layer_attention = attentions[m_layers]  # 第一层
    n_head_attention = m_layer_attention[0, n_heads]  # 第一个头的注意力权重

    plt.imshow(n_head_attention.detach().cpu(), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Map - Layer 0, Head 0')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.show()


def test_visualize_attention_diff(model, input_ids, attention_mask, layer_index, head_index):
    # 屏蔽之前的注意力
    print("屏蔽之前的注意力图：")
    visualize_attention(model, attention_mask, input_ids)

    # 屏蔽指定的 Attention Head
    model = knock_out_nth_layer_mth_attention_head(model, layer_index, head_index)
    
    # 屏蔽之后的注意力
    print("屏蔽之后的注意力图：")
    visualize_attention(model, attention_mask, input_ids)