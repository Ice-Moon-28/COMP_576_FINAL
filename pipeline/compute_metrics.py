import glob
import json
import os
import pickle

import setting
from util.get_auc_score import getAUROC




def main(args):
    print(args)
    old_sequences = []
    model_name = args.model
    # if '/' in model_name:
    #     model_name = model_name.replace('/', '_')
    cache_dir = os.path.join(setting.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
    os.makedirs(cache_dir, exist_ok=True)
    old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
    print(old_results)
    old_results = [_ for _ in old_results if '_partial' not in _]
    if len(old_results) > 0:
        getAUROC(file_name=old_results[0], args=args)
    run_id = len(old_results)
    with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
        json.dump(args.__dict__, f)