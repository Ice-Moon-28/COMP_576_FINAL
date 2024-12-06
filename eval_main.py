import torch
from pipeline.compute_metrics import main
from pipeline.pipeline_parser import args

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# 然后继续你的代码
if __name__ == '__main__':
    # try:
    task_runner = main(args=args)

    # except Exception as e:
    #     print(f"Error occurred: {e}")
