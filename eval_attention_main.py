import torch
from pipeline.pipeline_parser import args

import os

from util.get_auc_score import getAttentionAUROC

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

if __name__ == '__main__':
    task_runner = getAttentionAUROC(file_name='items.pkl', args=args)
