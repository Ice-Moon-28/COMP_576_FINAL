import pickle
import numpy as np


prompts = []
Q = []
AnswerGT = []
Energy = []
MostLikelyAns = []
Perplexity = []
NormalizedEntropy = []
Batch_Generations = []
EigenValue_all_attentions = []
EigenScore_all_attentions = []
EigenValue_v3_attentions = []
EigenScore_v3_attentions = []



with open('output2.log', 'r') as file:
    lines = file.readlines()
    in_batch = False
    in_eigen = False
    eigen_score_buffer = []  # 临时缓存 EigenScore 的数据
    count = 0
    while len(lines) > 0:
        line = lines.pop(0)
        count += 1
        if count % 10000 == 0:
            print("Has Analysed : ", count)        
        # 提取不同字段
        if line.startswith("Prompt:"):
            prompts_buffer = []
            prompts_buffer.append(line.split("Prompt:", 1)[1].strip())
            while True:
                line = lines[0]
                if line.startswith("Question:"):
                    prompts.append("".join(prompts_buffer))
                    break
                else:
                    line = lines.pop(0)
                    count += 1
                    print(count, '2') 
                    prompts_buffer.append(line.strip())
        elif line.startswith("Question:"):
            Q.append(line.split("Question:", 1)[1].strip())
        elif line.startswith("AnswerGT:"):
            AnswerGT.append(line.split("AnswerGT:", 1)[1].strip())
        elif line.startswith("Energy:"):
            Energy.append(float(line.split("Energy:", 1)[1].strip()))
        elif line.startswith("MostLikelyAns:"):
            MostLikelyAns.append(line.split("MostLikelyAns:", 1)[1].strip())
        elif line.startswith("Perplexity:"):
            Perplexity.append(float(line.split("Perplexity:", 1)[1].strip()))
        elif line.startswith("NormalizedEntropy:"):
            NormalizedEntropy.append(float(line.split("NormalizedEntropy:", 1)[1].strip()))
        
        # 解析 Batch_Generations 多行数据
        elif line.startswith("Batch_Generations:"):
            Batch_Generations.append(line.split("Batch_Generations:", 1)[1].strip())
        # 解析 EigenScore-all-attentions 并根据 `])]]` 结束符判断结束
        elif line.startswith("EigenValue-all-attentions:"):
            eigen_score_buffer = []
            eigen_score_buffer.append(line.split("EigenValue-all-attentions:", 1)[1].strip())
            while True:
                line = lines.pop(0)
                count += 1
                print(count)
                if line.endswith("])]]\n"): 
                    eigen_score_buffer.append(line.strip())
                    EigenValue_all_attentions.append(np.array(eval("".join(eigen_score_buffer), {"array": np.array, "np": np})))# 保存缓冲区的结果
                    eigen_score_buffer = []
                    break
                else:
                    eigen_score_buffer.append(line.strip()) 
        elif line.startswith("EigenScore-all-attentions:"):
            EigenScore_all_attentions.append(np.array(eval(line.split("EigenScore-all-attentions:", 1)[1].strip())))
            pass
        elif line.startswith("EigenValue-v3-attention:"):
            eigen_score_buffer = []
            eigen_score_buffer.append(line.split("EigenValue-v3-attention:", 1)[1].strip())
            while True:
                line = lines.pop(0)
                count += 1
                print(count, '1') 
                if line.endswith("])]\n"): 
                    eigen_score_buffer.append(line.strip())
                    EigenValue_v3_attentions.append(np.array(eval("".join(eigen_score_buffer), {"array": np.array, "np": np}))) # 保存缓冲区的结果
                    eigen_score_buffer = []  # 清空缓存
                    break
                else:
                    eigen_score_buffer.append(line.strip()) 
        elif line.startswith("EigenScore-v3-attention:"):
            EigenScore_v3_attentions.append(np.array(eval(line.split("EigenScore-v3-attention:", 1)[1].strip())))

print(len(prompts))
print(len(Q))
print(len(AnswerGT))
print(len(Energy))
print(len(MostLikelyAns))
print(len(Perplexity))
print(len(NormalizedEntropy))
print(len(Batch_Generations))
print(len(EigenValue_all_attentions))
print(len(EigenScore_all_attentions))
print(len(EigenValue_v3_attentions))
print(len(EigenScore_v3_attentions))


items = []

for i in range(len(EigenValue_all_attentions)):
    item = {}
    item["prompt"] = prompts[i]
    item["question"] = Q[i]
    item["answer"] = AnswerGT[i]
    item["energy"] = Energy[i]
    item["most_likely_generation"] = MostLikelyAns[i]
    item["perplexity"] = Perplexity[i]
    item["normalized_entropy"] = NormalizedEntropy[i]
    item["batch_generations"] = Batch_Generations[i]
    item["eigenValue_all_attentions"] = EigenValue_all_attentions[i]
    item["eigenScore_all_attentions"] = EigenScore_all_attentions[i]
    item["eigenValue_v3_attentions"] = EigenValue_v3_attentions[i]
    item["eigenScore_v3_attentions"] = EigenScore_v3_attentions[i]
    item["id"] = i

    items.append(item)


def save_pickle():
    with open("items.pkl", "wb") as f:
        pickle.dump(items, f)

save_pickle()
