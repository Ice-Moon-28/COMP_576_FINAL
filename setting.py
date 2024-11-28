import getpass
import os
import sys
import torch

__USERNAME = getpass.getuser()
_BASE_DIR = f'.'
MODEL_PATH = f'{_BASE_DIR}/weights/'
DATA_FOLDER = os.path.join(_BASE_DIR, 'data')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)


SENTENCES_TRANSFORMER_PATH = f'{_BASE_DIR}/weights/nli-roberta-large'
BERT_SCORE = f'{_BASE_DIR}/weights/bert_score'
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"