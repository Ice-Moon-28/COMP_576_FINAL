from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.coqa import get_dataset

# 初始化模型和 tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # 替换为你的 LLAMA 模型路径
tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir="./weights/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir="./weights/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token  # 确保 pad_token 设置正确

model = model.to("mps")

# 加载 CoQA 数据集
def preprocess(example):
    # 构造 Prompt 格式：Story + 问题
    example["prompt"] = example["story"] + " Q: " + example["question"] + " A:"
    return example

dataset = get_dataset(tokenizer=tokenizer)

# 定义 collate_fn
def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    inputs = tokenizer(prompts, truncation=True, padding=True, return_tensors="pt")
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "original_batch": batch  # 保留原始数据
    }

# 创建 DataLoader
batch_size = 2 # 根据硬件调整 batch_size
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# 推理和生成
model.eval()
all_results = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Generating answers"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # max_length=512,  # 根据需要调整最大生成长度
            num_beams=1,  # 如果需要多样性，可以增加
            do_sample=False,  # 是否使用采样
            pad_token_id=tokenizer.pad_token_id,  # 指定填充 token
            eos_token_id=tokenizer.eos_token_id  # 指定终止 token
        )

        # 解码生成的序列
        generations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # 逐个对齐生成的答案与原始问题
        for idx, generation in enumerate(generations):
            original_item = batch["original_batch"][idx]
            all_results.append({
                "id": original_item["id"],
                "story": original_item["story"],
                "question": original_item["question"],
                "gold_answer": original_item["answer"],
                "generated_answer": generation
            })

            print({
                "id": original_item["id"],
                "story": original_item["story"],
                "question": original_item["question"],
                "gold_answer": original_item["answer"],
                "generated_answer": generation
            })

# 保存或打印结果
import json
with open("coqa_generated_answers.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("生成完成，结果已保存至 coqa_generated_answers.json")