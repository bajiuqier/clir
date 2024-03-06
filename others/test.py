#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import ir_measures
from ir_measures import * 
from functools import partial
from tqdm.auto import tqdm
import json
import subprocess
import os
import glob
import sys
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import time
import datetime
import argparse
logging.disable(logging.INFO)
logging.disable(logging.WARNING)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


METRICS_LIST = [AP@5,  AP@10, AP@100, R@5, R@10, R@100, nDCG@5, nDCG@10, nDCG@100, Rprec]


def format_time(elapsed_time):
    elapsed_rounded = int(round((elapsed_time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def encode_text(model, max_seq_length, tokenizer, text):    
    input = tokenizer.batch_encode_plus(
            text,
            truncation="longest_first", 
            max_length = max_seq_length, 
            padding="max_length", 
            add_special_tokens=True, 
            return_tensors="pt", 
            return_attention_mask=True, 
            return_token_type_ids=False,
            )
    input = { k: v.to(device) for k, v in input.items() }
    return model(**input)


def run_inference(args, ckpt_filename, save_irmeasures_rank_filename):
    # ----------------- 1. loading model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir=args.cache_dir)

    lm_model = AutoModel.from_pretrained(os.path.join(ckpt_filename, 'lm_model'), cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    lm_model.to(device)
    print("... Model loaded into GPU")   
 

    # ----------------- 2. inference
    # partial是python中的一个高阶函数,作用是部分应用一个函数。即固定一个函数的部分参数,返回一个新的函数。
    # 这里固定了 encode_text 需要的 lm_model, args.max_seq_length, tokenizer 三个参数，并生成了一个新的函数 text_encoder
    text_encoder = partial(encode_text, lm_model, args.max_seq_length, tokenizer)
   

    # ----------------- 3. loading test data
    test_df = pd.read_csv(args.testinput_filename)
    test_df['qid'] = test_df['qid'].astype(str)
    print("... Test samples : ", test_df.shape[0])


    # ----------------- 4. do inference
    min_fD_rank = 0
    max_fD_rank = min_fD_rank + args.num_feedback_docs - 1
    print(f"... using {args.num_feedback_docs} number of feedbacks")
    print(f"... using {args.weighting_mode} averaging")
    lm_model.eval()
    with torch.no_grad():
        res = []
        for (qid, query), df in tqdm(test_df.groupby(['qid', 'query']), desc='Ranking'):
            scores = []

            # -------------------------------- Feeding a set of feedbackDocs with weights
            feedbackDoc_list = df[df['feedback'] == 1][ (min_fD_rank <= df['rank']) & (df['rank'] <= max_fD_rank) ].passage.tolist()

            Q_ = text_encoder([[query, feedbackDoc] for feedbackDoc in feedbackDoc_list])['last_hidden_state'][:, 0, :]
        

            weights = None
            if args.weighting_mode == 'bm25weights':
                weights = df[df['feedback'] == 1][ (min_fD_rank <= df['rank']) & (df['rank'] <= max_fD_rank) ]['bm25_score'].tolist()
            elif args.weighting_mode == 'rr':
                weights = df[df['feedback'] == 1][ (min_fD_rank <= df['rank']) & (df['rank'] <= max_fD_rank) ]['rank'].tolist()        
                weights = [1/(e+1) for e in weights]  
                    
            if weights:
                weights = torch.FloatTensor(weights).to(device)
                Q = weights.T @ Q_ / weights.sum()
            else:
                Q = torch.mean(Q_, dim=0, keepdim=True)


            for i in tqdm(range(0, df.shape[0], args.batch_size), leave=False):

                D_ = text_encoder(df.iloc[i: i+args.batch_size].passage.tolist())['last_hidden_state'][:, 0, :]

                batch_scores = (D_ @ Q.T).view(-1).to('cpu')          
                    
                scores += batch_scores.tolist()

            assert len(scores) == df.shape[0]
            res.append( df.assign(score=scores) )

    run = pd.concat(res).groupby(['qid', 'docid'])\
                                    .score.max().groupby('qid').agg(lambda x: x.droplevel('qid').to_dict()).to_dict()


    # ----------------- 5. save results in json
    with open(save_irmeasures_rank_filename, 'w') as fw:
        json.dump(run, fw, indent=4)


    # ----------------- 6. trec_eval computatation using ir-measures library
    qrels = list(ir_measures.read_trec_qrels(args.qrle_filename))
    res = ir_measures.calc_aggregate(METRICS_LIST, qrels, run) 

    return res


def main_run(args):
    t1 = time.time()


    eval_dir = args.results_dir
    # os.makedirs(path, exist_ok=True)可以保证无论目录是否存在,都不会抛出FileExistsError的异常,程序可以继续执行。
    # exist_ok=True的参数所起的作用是:当目录路径已存在时,不抛出异常。
    os.makedirs(eval_dir, exist_ok=True)


    ckpt_filenames = []
    if args.single == 'True':
        ckpt_filenames = [args.finetuned_model_dir]
        print("Single ckpt file found : ", args.finetuned_model_dir)
    else:   
        # 获取 fine-tune 模型的所有 checkpoint 文件路径     
        ckpt_filenames = sorted(glob.glob(os.path.join(args.finetuned_model_dir, "savechpt-*")))
        print("Looping over ckpt files : ", ckpt_filenames)


    final_scores_dict = {}
    for i, ckpt_filename in enumerate(ckpt_filenames):
        # 从checkpoint文件的完整路径中提取出文件名(不含目录)。
        # os.path.basename函数可以从一个路径中提取出最后的文件名或目录名部分。
        ckpt_postfix = os.path.basename(ckpt_filename)
        save_irmeasures_rank_filename = Path(eval_dir) /  f"{args.weighting_mode}.ranked_list.{ckpt_postfix}.{args.passage_or_doc}.json"
        

        per_ckpt_eval_measures = run_inference(args, ckpt_filename=ckpt_filename, save_irmeasures_rank_filename=save_irmeasures_rank_filename)
        

        print("Finished inference on checkpoint : ", ckpt_postfix)
        formatted_scores = {str(k): v for k, v in per_ckpt_eval_measures.items()}
        final_scores_dict[ckpt_postfix] = formatted_scores
       
        final_scores_df = pd.DataFrame( final_scores_dict )
        final_scores_df.to_csv(os.path.join(eval_dir, f"{args.weighting_mode}_aggregated_eval_scores_{args.passage_or_doc}.csv"), index=True)


    elapsed = format_time(time.time() - t1)
    print('Completed the inference : {:}'.format(elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('-s', '--single', type=str, default='True')
    parser.add_argument('-f', '--finetuned_model_dir', type=str, required=True)
    parser.add_argument('-fd', '--num_feedback_docs', type=int, required=True)
    parser.add_argument('-w', '--weighting_mode', type=str, default='uniform')
    parser.add_argument('-p', '--passage_or_doc', type=str, required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=5)
    parser.add_argument('-la', '--lang', type=str, default='fa')
    parser.add_argument('-r', '--results_dir', type=str, default='./eval')
    parser.add_argument('-t', '--testinput_filename', type=str, default='./fa.title.test.bm25.fullText.csv')
    parser.add_argument('-ms', '--max_seq_length', type=int, default=180)

    # default flags
    parser.add_argument('-c', '--cache_dir', type=str, default='./cache_dir/')
    parser.add_argument('-l', '--local_files_only', type=bool, default=True)
    parser.add_argument('-m', '--model_type', type=str, default='xlm-roberta-base')
    parser.add_argument('-q', '--qrle_filename', type=str, default='./all.qrels.txt')
    args = parser.parse_args()
    print(args, '\n')


    main_run(args)

