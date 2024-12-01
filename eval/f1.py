import re
import string

def normalize_answer(s):
    """
    标准化答案：去除标点符号、文章冠词，并将大写转为小写
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_f1(prediction, ground_truth):
    """
    计算 F1-score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    num_common = len(common_tokens)

    if num_common == 0:
        return 0, 0, 0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall

def compute_exact_match(prediction, ground_truth):
    """
    计算 Exact Match (EM)
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_f1_em(predictions, references):
    """
    对 CoQA 数据集进行评分
    predictions: 模型生成的答案列表
    references: 参考答案列表
    """
    total_f1, total_em, total_precision, total_recall = 0, 0, 0 ,0
    for pred, ref in zip(predictions, references):
        f1, precision, recall = compute_f1(pred, ref)
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        total_em += compute_exact_match(pred, ref)

    avg_f1 = total_f1 / len(references)
    avg_em = total_em / len(references)
    avg_precision = total_precision / len(references)
    avg_recall = total_recall / len(references)
    return {"F1-score": avg_f1,  "Exact Match": avg_em, "Precision": avg_precision, "Recall": avg_recall}


if __name__ == "__main__":
    predictions = ["this is an answer", "another answer"]
    references = ["this is the answer", "another answer"]

    results = evaluate_f1_em(predictions, references)
    print("Evaluation Results:", results)