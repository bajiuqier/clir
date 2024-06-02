import pandas as pd
import ir_datasets
import ast
from pathlib import Path
from tqdm import tqdm
import os
import jsonlines

from event_reminder import SendEmail

'''
CLIRMatrix zk kk BI-139 full 数据
为 zh-query 构建一个相关性最高的正样本 kk文档 随机构建一个 负样本
导出为 jsonl 数据 存储的是 query 和 document 的文本形式
query pos neg
'''

def reformatted_qrels(dataset, data_dir: str, batch_size: int=500000, pos_doc_num: int=1, neg_doc_num: int=1) -> None:
    qrels_df = pd.DataFrame(dataset.qrels_iter())
    grouped_qid = qrels_df.groupby('query_id')

    # 初始化一个空列表来存储重构后的数据
    reformatted_qrels_list = []
    
    epoch = 1

    # 遍历每个查询ID
    # for query_id, group in tqdm(grouped_qid, total=len(grouped_qid.size())):
    for query_id, group in grouped_qid:
        # 获取当前查询的所有相关文档
        # 从 group 中获取doc_id和relevance这两列数据,并按照relevance列的值进行降序排序。
        relevant_docs = group[['doc_id', 'relevance']].sort_values(by='relevance', ascending=False)
        
        # 提取前pos_doc_num个正样本
        positive_docs = relevant_docs.head(pos_doc_num)
        
        # 从剩余的所有文档中随机选择n个作为负样本
        # 负样本 -> 除去query对应的文档（无论对应的文档的相似度是多少，没有相似度为0的文档）其他所有文档的id
        negative_docs = qrels_df[~qrels_df['doc_id'].isin(relevant_docs['doc_id'])]['doc_id'].sample(neg_doc_num)
        # 一个优化方案：取出所有 doc_id 并去重，然后在所有doc_id中随机选择 neg_doc_num 个，要求这两个 doc_id 不可以是当前 query_id 对应的 doc_id

        # 将query_id pos_docs_id 和 neg_docs_id 组合成一个新的 list
        reformatted_example = [str(query_id), positive_docs['doc_id'].tolist(), negative_docs.tolist()]
        # 将重构后的数据添加到列表中
        reformatted_qrels_list.append(reformatted_example)

        reformatted_qrels_file = os.path.join(data_dir, f'reformatted_qrels_{epoch}.csv')

        # 每一个 batch_size 存储一下数据
        if len(reformatted_qrels_list) >= batch_size:
            # 将重构后的数据组合成一个DataFrame
            reformatted_qrels_df = pd.DataFrame(reformatted_qrels_list, columns=['query_id', 'pos_docs_id', 'neg_docs_id'])
            if epoch == 1:
                reformatted_qrels_df.to_csv(reformatted_qrels_file, mode='a', index=False)
            else:
                reformatted_qrels_df.to_csv(reformatted_qrels_file, mode='a', header=False, index=False)
            reformatted_qrels_list = []

        epoch = epoch + 1
                
    if reformatted_qrels_list:
            reformatted_qrels_df = pd.DataFrame(reformatted_qrels_list, columns=['query_id', 'pos_docs_id', 'neg_docs_id'])
            reformatted_qrels_df.to_csv(reformatted_qrels_file, mode='a', header=False, index=False)

    return None



def get_train_data(dataset, reformatted_qrels_file: str, train_data_file: str) -> None:

    docstore = dataset.docs_store()
    queries_df = pd.DataFrame(dataset.queries_iter())

    reformatted_qrels_df = pd.read_csv(reformatted_qrels_file, encoding='utf-8')
    if not isinstance(reformatted_qrels_df.loc[0]['query_id'], str):
        reformatted_qrels_df['query_id'] = reformatted_qrels_df['query_id'].astype(str)

    example = {'query':'', 'pos':'', 'neg':''}

    with open(train_data_file, 'w', encoding='utf-8') as f:
        writer = jsonlines.Writer(f)
        # for index, row in tqdm(reformatted_qrels_df.iterrows(), total=reformatted_qrels_df.shape[0]):
            
        for index, row in reformatted_qrels_df.iterrows():
            print('index:', index)
            assert type(row['query_id']) == str, '数据中的 query_id 的类型不是 str 请检查'

            q_text = queries_df.loc[queries_df['query_id'] == row['query_id'], 'text'].values[0]

            pos_docs_id = ast.literal_eval(row['pos_docs_id'])
            assert type(pos_docs_id) == list, 'pos_docs_id 不是一个 list 类型 请检查'
            neg_docs_id = ast.literal_eval(row['neg_docs_id'])
            assert type(neg_docs_id) == list, 'neg_docs_id 不是一个 list 类型 请检查'
            pos_docs_text = [doc.text for doc in docstore.get_many_iter(pos_docs_id)]
            neg_docs_text = [doc.text for doc in docstore.get_many_iter(neg_docs_id)]

            example['query'] = q_text
            example['pos'] = pos_docs_text
            example['neg'] = neg_docs_text
            writer.write(example)

            # 发送邮件提醒
            if index+1 % 50000 == 0:
                subject = '任务进度通知'
                content = f'已经完成-{index}。'
                email = SendEmail(subject, content)
                email.send_mail()

            if index == 10:
                break

    return None


if __name__ == '__main__':

    HOME_DIR = Path(__file__).parent
    reformatted_qrels_file = str(HOME_DIR / 'reformatted_qrels.csv')
    train_data_file = str(HOME_DIR / 'train_data.jsonl')

    dataset = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train')

    # reformatted_qrels(dataset=dataset, data_dir=HOME_DIR, pos_doc_num=2, neg_doc_num=2)

    get_train_data(dataset=dataset, reformatted_qrels_file=reformatted_qrels_file, train_data_file=train_data_file)

    # 发送邮件提醒
    # subject = '任务进度通知'
    # content = '您的任务已经完成，请查收。'
    # email = SendEmail(subject, content)
    # email.send_mail()


    
