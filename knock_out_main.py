import torch
from pipeline.knock_out_pipeline import knock_out_main as main
from pipeline.pipeline_parser import args

if __name__ == '__main__':
    # try:
    task_runner = main(parallel=args.nprocess)

    # except Exception as e:
    #     print(f"Error occurred: {e}")