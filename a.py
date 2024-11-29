from transformers import AutoTokenizer, AutoModel
from torchmetrics.text.bert import BERTScore


bertscore = BERTScore(model_name_or_path="bert-base-uncased", device='cuda')