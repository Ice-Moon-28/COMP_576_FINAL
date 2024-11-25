from transformers import AutoModel, AutoTokenizer

# Specify cache location
cache_dir = "./weights/Llama-2-7b-hf"

# Download model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)