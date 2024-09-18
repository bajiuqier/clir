import os
import argparse
import subprocess
import random
import tempfile
from tqdm import tqdm
import torch
import modeling
import data
import pytrec_eval
from statistics import mean
from collections import defaultdict

# 将输出写进日志文件
from datetime import datetime
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

SEED = 42
LR = 0.001
BERT_LR = 1e-5
MAX_EPOCH = 16
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 32
GRAD_ACC_SIZE = 2
#other possibilities: ndcg
# VALIDATION_METRIC = 'P_20'
# VALIDATION_METRIC = 'ndcg_cut'
VALIDATION_METRIC = {'ndcg_cut_5', 'ndcg_cut_10'}


PATIENCE = 20 # how many epochs to wait for validation improvement

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}


def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid, model_out_dir=None):
    '''
        Runs the training loop, controlled by the constants above
        Args:
            model(torch.nn.model or str): One of the models in modelling.py, 
            or one of the keys of MODEL_MAP.
            dataset: A tuple containing two dictionaries, which contains the 
            text of documents and queries in both training and validation sets:
                ({"q1" : "query text 1"}, {"d1" : "doct text 1"} )
            train_pairs: A dictionary containing query document mappings for the training set
            (i.e, document to to generate pairs from). E.g.:
                {"q1: : ["d1", "d2", "d3"]}
            qrels_train(dict): A dicationary containing training qrels. Scores > 0 are considered
            relevant. Missing scores are considered non-relevant. e.g.:
                {"q1" : {"d1" : 2, "d2" : 0}}
            If you want to generate pairs from qrels, you can pass in same object for qrels_train and train_pairs
            valid_run: Query document mappings for validation set, in same format as train_pairs.
            qrels_valid: A dictionary  containing qrels
            model_out_dir: Location where to write the models. If None, a temporary directoy is used.
    '''

    log_dir = str(Path.home().parent / 'mnt' / 'workspace' / 'clir' / 'training_logs')
    # 按日期命名日志文件
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"baseline_{model}_training_{current_date}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", # 这里设置时间格式
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    if isinstance(model,str):
        model = MODEL_MAP[model]().cuda()
    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    epoch = 0
    top_valid_score = None
    # print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)
    logger.info(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}')
    for epoch in range(MAX_EPOCH):

        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels_train)
        # print(f'train epoch={epoch} loss={loss}')
        logger.info(f'train epoch={epoch} loss={loss}')

        valid_score = validate(model, dataset, valid_run, qrels_valid, epoch)
        # print(f'validation epoch={epoch} score={valid_score}')
        logger.info(f'validation epoch={epoch} score={valid_score}')


        # if top_valid_score is None or valid_score > top_valid_score:
        if top_valid_score is None or mean(valid_score.values()) > mean(top_valid_score.values()):

            top_valid_score = valid_score
            # print('new top validation score, saving weights', flush=True)
            logger.info('new top validation score, saving weights')

            model.save(os.path.join(model_out_dir, 'weights.p'))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            # print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            logger.info(f'no validation improvement since {top_valid_score_epoch}, early stopping')

            break
        
    #load the final selected model for returning
    if top_valid_score_epoch != epoch:
        model.load(os.path.join(model_out_dir, 'weights.p'))
    return (model, top_valid_score_epoch)


def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, valid_qrels, epoch):
    run_scores = run_model(model, dataset, run)
    metric = VALIDATION_METRIC
    # if metric.startswith("ndcg_cut_"):
    #     metric = "ndcg_cut"
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, metric)
    eval_scores = trec_eval.evaluate(run_scores)
    # print(eval_scores)
    # return mean([d[VALIDATION_METRIC] for d in eval_scores.values()])
    scores = {}
    for METRIC in VALIDATION_METRIC:
        scores[METRIC] = mean([d[METRIC] for d in eval_scores.values()])
    return scores



def run_model(model, dataset, run, desc='valid'):
    rerank_run = defaultdict(dict)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    return rerank_run
    

def write_run(rerank_run, runf):
    '''
        Utility method to write a file to disk. Now unused
    '''
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')

    parser.add_argument('--train_qrels', type=argparse.FileType('rt'))
    parser.add_argument('--valid_qrels', type=argparse.FileType('rt'))
    
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')

    args = parser.parse_args()
    
    model = MODEL_MAP[args.model]().cuda()
    dataset = data.read_datafiles(args.datafiles)

    train_qrels = data.read_qrels_dict(args.train_qrels)
    valid_qrels = data.read_qrels_dict(args.valid_qrels)

    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    # we use the same qrels object for both training and validation sets
    main(model, dataset, train_pairs, train_qrels, valid_run, valid_qrels, args.model_out_dir)


if __name__ == '__main__':
    main_cli()
