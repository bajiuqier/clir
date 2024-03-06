import ir_datasets
import logging
from datetime import datetime
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path
import json
import csv
import os
import argparse

'''
`$ python3 train_data_download.py --outDir ./irds_out/ --dataDir ./ --clir_matrix_fname irds_list.txt --relevance_score 6`
'''

'''global variables'''
LOGGER = None
args = None

def get_dataset_examples(dataset_fname):

    # local result variables
    # collections是Python内置的一个集合模块,defaultdict:访问不存在的 key 时返回默认值 0
    queries_dict, docs_dict, training_examples = defaultdict(lambda: 0), defaultdict(lambda: 0), []

    # dataset_fname example: 'clirmatrix/zh/bi139-base/en/train'
    curr_dataset = ir_datasets.load(dataset_fname)

    # returns list of tuples: [(query_id, doc_id, relevance, iteration)]
    most_relevant_docs = get_list_of_most_relevant_docs(curr_dataset)

    # many-many relationship
    # two dict qid:qtext and docid:doctext (size of doc_ids is size of most_relevant_docs)
    query_ids, doc_ids, _, _ = zip(*most_relevant_docs)

    # create subDir
    output_dir = Path(args.outDir) / ('relevance_score' + str(args.relevance_score))
    output_dir.mkdir(exist_ok=True, parents=True)

    docs_dict = curr_dataset.docs.lookup(doc_ids)

    queries_dict = curr_dataset.queries.lookup(query_ids)

    # generate tuples
    for qid, did in zip(query_ids, doc_ids):
        training_examples.append((queries_dict[qid].text, docs_dict[did].text))

    print(f'ntraining_examples generated for {dataset_fname}: {len(training_examples)}')
    
    return training_examples


def write_examples_to_csv(dataset_fname, training_examples):

    # original: 'clirmatrix/zh/bi139-base/en/train'
    # replaced: 'clirmatrix_zh_bi139-base_en_train.csv'
    # 所以 dataset_fname.replace('/','_') 使用 '_' 代替 '/'
    csv_outfile = Path(args.outDir) / ('relevance_score' + str(args.relevance_score)) / (dataset_fname.replace('/','_') + '.csv')
    print(f'confirm csv_outfile: {csv_outfile}') 
    
    with open(csv_outfile, mode='w') as csv_file_run_id:
        # 具体参数说明:
        # csv_file_run_id:打开的csv文件对象,用于写入
        # fieldnames:列名列表,表示csv中每一列的标题
        # quoting:引用方式,csv.QUOTE_ALL表示全列引用,即所有字段用引号包裹
        writer = csv.DictWriter(csv_file_run_id, fieldnames=['query_text', 'doc_text'], quoting=csv.QUOTE_ALL)
        # writer.writeheader()
        for example in training_examples:
            query_text, doc_text = example[0], example[1]
            writer.writerow({'query_text': query_text, 'doc_text': doc_text})

def get_list_of_most_relevant_docs(dataset):
    return [ qrel 
        for qrel in tqdm(dataset.qrels, desc='reading qrels') 
        if qrel.relevance >= args.relevance_score 
    ]

'''
文件中并没有用到json_dump()和json_load()。
这两个方法可能是为了完整性而定义,方便之后扩展使用。
如果后续需要以json格式保存数据,可以直接调用json_dump()来实现,无需再次定义。
总结一下:
- json_dump()和json_load()已定义但未在本文件中使用
- 用于json格式的序列化/反序列化
- 可在未来扩展中启用这两个方法
'''
def json_dump(data, output_json_file):
    with open(output_json_file, 'w') as fout:
        json.dump(data, fout)

def json_load(filename):
    with open(filename, 'r') as fin:
        data = json.load(fin)
    fin.close()
    return data

def set_logger():
    """Helper function that formats a logger use programmers can easily debug their scripts.
    Args:
      N/A
    Returns:
      logger object
    Note:
      You can refer to this tutorial for more info on how to use logger: https://towardsdatascience.com/stop-using-print-and-start-using-logging-a3f50bc8ab0
    """

    # STEP 1
    # create a logger object instance
    # 创建一个logger对象,这是日志记录器的基础。
    logger = logging.getLogger()

    # STEP 2
    # specifies the lowest severity for logging
    # 设置logger的日志级别为最低级别NOTSET,即记录所有级别的日志。
    logger.setLevel(logging.NOTSET)

    # STEP 3
    # 创建两个handler来处理日志的输出:
        # 一个是输出到控制台的StreamHandler
        # 一个是输出到文件的FileHandler
    # set a destination for your logs or a "handler"
    # here, we choose to print on console (a consoler handler)
    console_handler = logging.StreamHandler()
    # here, we choose to output the console to an output file
    # file_handler = logging.FileHandler("mylog.log")

    # STEP 4
    # set the logging format for your handler
    # 设置日志的格式(log_format),包含了日期、行号、日志级别等信息。
    # 使用了格式化字符串语法,其中各个字段表示:

    # \n - 换行符
    # %(asctime)s - 日志输出时间
    # Line %(lineno)d - 日志输出代码行号
    # in %(filename)s - 日志输出代码文件名
    # %(funcName)s() - 日志输出函数名
    # %(levelname)s - 日志级别
    # %(message)s - 日志文本消息
    # 一个示例输出如下:
    # 2022-08-30 15:49:23,924 | Line 12 in train.py: train() | INFO: Start training model.

    log_format = '\n%(asctime)s | Line %(lineno)d in %(filename)s: %(funcName)s() | %(levelname)s: \n%(message)s'
    console_handler.setFormatter(logging.Formatter(log_format))

    # we add console handler to the logger
    # 添加handler到logger中, logger可以正确输出日志。
    logger.addHandler(console_handler)
    # we add file_handler to the logger
    # logger.addHandler(file_handler)

    return logger

def register_arguments():
    """Registers the arguments in the argparser into a global variable.
    Args:
      N/A
    Returns:
      N/A, sets the global args variable
    """

    global args

    # 创建解析对象
    parser = argparse.ArgumentParser()

    # Specify command line arguments.
    parser.add_argument(
        '--outDir', type=str,
        required=True,
        help="Name of output directory, this will help in keeping your runs organized."
        )
    parser.add_argument(
        '--dataDir', type=str,
        required=True,
        help="Name of data directory, tell script where to get input data files from."
        )
    parser.add_argument(
        '--clir_matrix_fname', type=str,
        required=True,
        help=".txt file containing a list of each of the CLIRmatrix datasets we are generating training pairs from."
        )
    parser.add_argument(
        '--relevance_score', type=int,
        required=True,
        help="Indicate the relevance score desired for training. Ex. 5 or 6"
        )

    # Parse command line arguments.
    # 调用parser的parse_args()方法解析参数,将结果存储到全局变量args中。
    args = parser.parse_args()

    # print command line arguments for this run
    # 打印出所有args中的参数,方便确认
    LOGGER.info("---confirm argparser---")
    for arg in vars(args):
        print(arg, getattr(args, arg))


def main():


    main_start = datetime.now()
    file_obj = open(args.dataDir + args.clir_matrix_fname, "r")
    
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    # 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
    clir_matrix_fnames = [fname.strip() for fname in file_obj.readlines()]

    LOGGER.info("list of datasets to generate: \n%s", clir_matrix_fnames)

    for dataset_fname in clir_matrix_fnames:

        LOGGER.info('Start Processing %s...', dataset_fname)

        curr_dataset_processing_start = datetime.now()
        curr_training_examples = get_dataset_examples(dataset_fname)
        write_examples_to_csv(dataset_fname, curr_training_examples)
        curr_dataset_processing_time = datetime.now() - curr_dataset_processing_start
        print(f'Finished processing {dataset_fname}.... computation time: {curr_dataset_processing_time}')
    main_time = datetime.now() - main_start
    LOGGER.info('Total time in main: %s', main_time)



if __name__ == '__main__':
    LOGGER = set_logger()
    register_arguments()
    main()