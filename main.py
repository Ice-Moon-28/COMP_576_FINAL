import torch
from pipeline.pipeline import main
from pipeline.pipeline_parser import args

if __name__ == '__main__':
    # try:
    task_runner = main(parallel=args.nprocess)

    # except Exception as e:
    #     print(f"Error occurred: {e}")