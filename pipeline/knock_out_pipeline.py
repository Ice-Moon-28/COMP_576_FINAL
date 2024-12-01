import argparse
import glob
import json
import os
import copy
import time

import pandas as pd
import torch
import tqdm
import transformers
from eval import coqa, squad
from eval.f1 import evaluate_f1_em
from log.log import get_knock_out_info
from sentence_transformers import SentenceTransformer
import models.load_model
import setting
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torchmetrics.text.bert import BERTScore

import setting
# import dataeval.coqa as coqa
# import dataeval.nq_open as nq_open
# import dataeval.triviaqa as triviaqa
# import dataeval.SQuAD as SQuAD
# import dataeval.TruthfulQA as TruthfulQA
import models
import util
from util.knock_out import knock_out_nth_layer_mth_attention_head, test_visualize_attention_diff
from util.metrics import get_lenghthNormalized_entropy, get_perplexity_score, getAvgBertScore, getEigenIndicator_v0, getEigenIndicatorOutput, getLexicalSim
from util.metrics import get_energy_score
from pipeline.pipeline_parser import args


logInfo = get_knock_out_info(args)


# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name, **args):
    dataset_map = {
        'coqa': lambda: coqa.get_dataset(**args),
        'squad': lambda: squad.get_dataset(**args),
        # 添加其他数据集时，可以以相同方式扩展
        # 'triviaqa': lambda: triviaqa.get_dataset(**args),
        # 'nq_open': lambda: nq_open.get_dataset(**args),
    }
    
    if data_name not in dataset_map:
        raise ValueError(f"Unsupported dataset name: {data_name}")
    
    return dataset_map[data_name]


def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    # if data_name == 'triviaqa':
    #     generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    # if data_name == 'nq_open':
    #     generation_config = nq_open._generate_config(tokenizer)
    if data_name == 'squad':
        generation_config = squad._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    # generation_config['pad_token_id'] = tokenizer.eos_token_id

    return generation_config
def collate_fn(batch, tokenizer):
    # 获取每个样本的 input_ids 和 attention_mask
    ids = [item['id'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    lengths = [len(item['input_ids']) for item in batch]

    prompts = [item["prompt"] for item in batch]
    inputs = tokenizer(prompts, truncation=True, padding=True, return_tensors="pt")
    
    return {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'id': ids,
        'question': questions,
        'answer': answers,
        'lengths': lengths,
        'original_batch': batch,
    }

def knock_out_process(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt, m_layers=0, n_heads=0, knock_out=True):
    device = args.device

    model= models.load_model.load_model(model_name=model_name, device=args.device)
    model.eval()

    tokenizer = models.load_model.load_tokenizer(model_name=model_name)

    util.seed_everything(seed)

    if knock_out:
        knock_out_nth_layer_mth_attention_head(model, m_layers, n_heads)

    dataset = get_dataset_fn(
        data_name=args.dataset,
        tokenizer=tokenizer,
    )()

    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch=batch, tokenizer=tokenizer),
    )

    sequences = []
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):

        input_ids = batch['input_ids'].to(device)

        # import pdb; pdb.set_trace()

        # test the knock out's effect
        # test_visualize_attention_diff(
        #     model,
        #     batch['input_ids'].to(device),
        #     batch['attention_mask'].to(device),
        #     m_layers,
        #     n_heads,
        # )

        input_lengths = batch['lengths']
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            dict_outputs = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                        num_beams=1,
                                        do_sample=False,
                                        generation_config=generation_config,
                                        output_hidden_states = True,
                                        return_dict_in_generate=True,
                                        output_scores=False,
                                    )
            # import pdb; pdb.set_trace()

            #
          
            most_likely_generations = [tokenizer.decode(seq[input_length:]) for seq, input_length in zip(dict_outputs.sequences.cpu(), input_lengths)]
        def print_log(batch_item, generation):
            input_ids = batch_item['input_ids']
            curr_seq = dict(
                prompt=tokenizer.decode(input_ids.cpu(), skip_special_tokens=True),
                id=batch_item['id'],
                question=batch_item['question'],
                answer=batch_item['answer'],
            )

            best_generated_text = generation

            scores = evaluate_f1_em(predictions=best_generated_text, references=batch_item['answer'])

            # {"F1-score": avg_f1,  "Exact Match": avg_em, "Precision": avg_precision, "Recall": avg_recall}
            
            print("Prompt:", tokenizer.decode(input_ids.cpu(), skip_special_tokens=True))
            print("Question:", batch_item['question'])
            print("AnswerGT:", batch_item['answer'])
            print("Best generated text:", best_generated_text)
            print("F1-score", scores['F1-score'])
            print("exact match", scores['Exact Match'])
            print("precision", scores['Precision'])
            print("recall", scores['Recall'])
            print("F1-score", scores['F1-score'], file=logInfo)
            print("exact match", scores['Exact Match'], file=logInfo)
            print("precision", scores['Precision'], file=logInfo)
            print("recall", scores['Recall'], file=logInfo)

            print(scores, file=logInfo)
            
            sequences.append(curr_seq)

        for i in range(len(most_likely_generations)):
            batch_item = {
                'input_ids': batch['input_ids'][i],
                'attention_mask': batch['attention_mask'][i],
                'id': batch['id'][i],
                'question': batch['question'][i],
                'answer': batch['answer'][i],
            }

            generation = most_likely_generations[i]

            print_log(batch_item, generation)

        if setting.device == 'cuda':
            torch.cuda.empty_cache()
        elif setting.device == 'mps':
            torch.mps.empty_cache()

    return sequences


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens


def knock_out_main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = util.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        # if '/' in model_name:
        #     model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(setting.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
            
    # print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    # print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    if args.knock_out:
        device = args.device

        model= models.load_model.load_model(model_name=model_name, device=args.device)
        num_layers = model.config.num_hidden_layers
        num_attention_heads = model.config.num_attention_heads
        del model
        print(f"模型的层数: {num_layers}") 
        for i in range(num_layers):
            for j in range(num_attention_heads):     
                sequences = knock_out_process(
                    model_name,
                    args, seed=args.seed,
                    old_sequences=old_sequences,
                    knock_out=True,
                    m_layers=i,
                    n_heads=j,
                )
                print(f'Writing {len(sequences)} generations to {cache_dir}...')
                pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}_{i}_{j}.pkl'))
    else:
        sequences = knock_out_process(model_name, args, seed=args.seed, old_sequences=old_sequences, knock_out=False)
        print(f'Writing {len(sequences)} generations to {cache_dir}...')
        pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return


