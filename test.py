from transformers import AutoModel, AutoTokenizer
import torch

# # Specify cache location
cache_dir = "./weights/Llama-2-7b-hf"

# # # Download model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16)

# print(model, tokenizer)

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("roberta-large-mnli", cache_dir="./weights/nli-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", cache_dir="./weights/nli-roberta-large")

# from bert_score import BERTScorer

# from models import load_model

# load_model.load_model(model_name="meta-llama/Llama-2-7b-hf", device='mps', torch_dtype=torch.float16)

# scorer = BERTScorer(model_type="bert-base", device="mps")

from transformers import BertModel, BertTokenizer

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('nli-roberta-large', cache_folder='./weights/nli-roberta-large')
model.save('./weights/nli-roberta-large')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./weights/bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased', cache_dir='./weights/bert-base-uncased')