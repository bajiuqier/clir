import logging
import pandas as pd
from pathlib import Path
import ir_datasets
import random
import jsonlines
from typing import NamedTuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def filter_qrels(qrels_df: pd.DataFrame, filter_reference_file: str) -> pd.DataFrame:

    # 加载 过滤 所需的文件
    filter_reference_df = pd.read_csv(filter_reference_file, encoding='utf-8').astype(str)
    query_ids = set(filter_reference_df['query_id'])

    # 过滤掉多余的 query_id 的文档数据
    qrels_filtered_df = qrels_df[qrels_df['query_id'].isin(query_ids)]
    # 删除 iteration 列数据
    qrels_filtered_df.drop('iteration', axis=1, inplace=True)

    return qrels_filtered_df

def build_new_base_train_qrels(original_qrels: pd.DataFrame, new_qrels_file: str=None, neg_doc_num: int=5, save_new_qrels: bool=True) -> pd.DataFrame:
    '''
    base 版本数据的 qrels 中，作者已经为每个 query 构建了 100 个对应的文档
    其中 相关度从高到底 6 5 4 3 2 1 0
    full 版本数据的 qrels 和 base不同的是 query 没有对应的负样本文档 及 没有相关度为 0 的文档 id
    '''
    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)

    # 计算每个 query_id 对应的 relevance 为 0 的 doc_id 数量
    neg_samples_counts_df = original_qrels[original_qrels['relevance'] == 0].groupby('query_id').size().reset_index(name='neg_sample_count')
    # query_id 对应的 relevance 为 0 的 doc_id 数量 不足 5个的 query_id
    insufficient_neg_samples_df = neg_samples_counts_df[neg_samples_counts_df['neg_sample_count'] < neg_doc_num]
    # 获取 这些 query_ids
    query_ids = insufficient_neg_samples_df['query_id'].unique()
    # 准备一个包含所有doc_id的列表，用于随机选择
    all_doc_ids = original_qrels['doc_id'].unique()

    # 定义一个列表 存储 添加的负样本
    new_rows = []

    for query_id in tqdm(query_ids, total=len(query_ids)):
        # 获取当前 query_id 对应的行
        current_docs = original_qrels[original_qrels['query_id'] == query_id]
        # 获取当前 query_id 下相关度为 0 的 doc_id
        zero_relevance_docs = current_docs[current_docs['relevance'] == 0]['doc_id']

        # 如果数量少于5，补足缺少的 doc_id
        if len(zero_relevance_docs) < neg_doc_num:
            # 当前缺少的 doc_id 数量
            shortage = neg_doc_num - len(zero_relevance_docs)
            
            # 获取当前 query_id 下所有的 doc_id
            current_doc_ids = current_docs['doc_id'].unique()

            # 从其他查询中随机选择不足的数量的文档ID
            # 并确保随机选择的文档ID不会与当前查询下的文档ID重复
            candidates_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in current_doc_ids]
            
            # 随机选择 shortage 数量的 doc_id
            selected_docs = random.sample(list(candidates_doc_ids), shortage)
            
            # 记录新添加的数据
            new_rows.extend([{'query_id': query_id, 'doc_id': doc, 'relevance': 0} for doc in selected_docs])

            # 如果添加了相关度为0的文档，则删除相关度最小的文档ID
            # 获取需要删除的行
            docs_to_remove = current_docs.iloc[:100 - len(zero_relevance_docs)].tail(shortage)
            # 删除这些行
            original_qrels = original_qrels.drop(docs_to_remove.index)

    if len(new_rows) > 0:
        original_qrels = pd.concat([original_qrels, pd.DataFrame(new_rows)], ignore_index=True)

    # 按 query_id 和 relevance 排序，保证每个 query_id 对应的100个 doc_id 是连续的
    original_qrels['query_id'] = original_qrels['query_id'].astype(int)
    original_qrels['relevance'] = original_qrels['relevance'].astype(int)
    new_qrels = original_qrels.sort_values(by=['query_id', 'relevance'], ascending=[True, False]).reset_index(drop=True)

    new_qrels['query_id'] = new_qrels['query_id'].astype(str)
    new_qrels['doc_id'] = new_qrels['doc_id'].astype(str)
    new_qrels['relevance'] = new_qrels['relevance'].astype(int)

    if save_new_qrels:
        new_qrels.to_csv(new_qrels_file, index=False, encoding='utf-8')
        print("------------------------------------------------------")
        print(f"新的 qrels 文件已经保存在{new_qrels_file}")
        print("------------------------------------------------------")
        
    return new_qrels

def build_dataset_old(
        docstore: NamedTuple,
        query_qid_file: str,
        qrels_file: str,
        item_info_file: str,
        adj_item_info_file: str,
        triple_id_file: str,
        output_file: str,
        adj_item_num: int=3,
        dataset_type: str="train",
        pos_doc_num: int=1,
        neg_doc_num: int=1
):

    '''
    构建数据集使用的 jsonl 数据    
    '''
    query_qid_df = pd.read_csv(query_qid_file, encoding='utf-8').astype(str)
    item_info_df = pd.read_csv(item_info_file, encoding='utf-8').astype(str)
    adj_item_info_df = pd.read_csv(adj_item_info_file, encoding='utf-8').astype(str)
    triple_id_df = pd.read_csv(triple_id_file, encoding='utf-8').astype(str)

    qrels_df = pd.read_csv(qrels_file, encoding='utf-8')
    qrels_df['query_id'] = qrels_df['query_id'].astype(str)
    qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
    qrels_df['relevance'] = qrels_df['relevance'].astype(int)

    # 构建JSONL数据
    jsonl_data = []

    for _, row in tqdm(query_qid_df.iterrows(), total=query_qid_df.shape[0], desc="building dataset"):
        query_id = row['query_id']
        query_text = row['query_text']
        q_item_qid = row['q_item_qid']

        # 获取查询对应实体的信息
        q_item = item_info_df[item_info_df['item_qid'] == q_item_qid]

        if q_item.shape[0] == 0:
            continue
        else:
            q_item = q_item.iloc[0]

        # 检查q_item中的'label_zh', 'label_kk' description_zh description_kk 是否有一个为空
        if q_item[['label_zh', 'label_kk', 'description_zh', 'description_kk']].isnull().any():
            continue

        q_item_info = {
            "label_zh": q_item['label_zh'],
            "label_kk": q_item['label_kk'],
            "description_zh": q_item['description_zh'],
            "description_kk": q_item['description_kk']
        }
        
        # 获取相邻实体的信息
        # adj_items = filtered_triplet_id[filtered_triplet_id['item'] == qid]['adjItem'].unique()
        adj_item_qids = triple_id_df[triple_id_df['item_qid'] == q_item_qid]['adj_item_qid']

        # 舍弃相邻实体数量为0的query
        if len(adj_item_qids) == 0:
            continue
        elif len(adj_item_qids) > 0 and len(adj_item_qids) < adj_item_num:
            adj_item_qids = adj_item_qids.tolist()
            while len(adj_item_qids) < adj_item_num:
                adj_item_qids.append(adj_item_qids[(adj_item_num - len(adj_item_qids)) % len(adj_item_qids)])
            adj_item_qids = pd.Series(adj_item_qids)
        else:
            # replace=False 是 pandas sample() 方法的一个参数，表示在抽样时不进行重复抽样
            # sampled_adj_items = adj_items.sample(adjitem_num, replace=False)
            adj_item_qids = adj_item_qids.sample(adj_item_num)

        adj_item_info = {
            "label_zh": [],
            "label_kk": [],
            "description_zh": [],
            "description_kk": []
        }
        
        stop_inner = False

        for adj_item_qid in adj_item_qids:
            adj_item = adj_item_info_df[adj_item_info_df['item_qid'] == adj_item_qid]

            if adj_item.shape[0] == 0:
                stop_inner = True
                break
            else:
                adj_item = adj_item.iloc[0]

            # 检查 adj_item 中的'label_zh', 'label_kk' description_zh description_kk 是否有一个为空
            if adj_item[['label_zh', 'label_kk', 'description_zh', 'description_kk']].isnull().any():
                stop_inner = True
                break

            adj_item_info["label_zh"].append(adj_item['label_zh'])
            adj_item_info["label_kk"].append(adj_item['label_kk'])
            adj_item_info["description_zh"].append(adj_item['description_zh'])
            adj_item_info["description_kk"].append(adj_item['description_kk'])
        
        if stop_inner:
            # 跳过外层循环中剩下的代码，进行下一次迭代
            continue
        
        if dataset_type == "train":
            query_docs = qrels_df[qrels_df['query_id'] == query_id]
            if query_docs.shape[0] == 0:
                continue
            else:
                pos_doc_ids = query_docs[query_docs['relevance'] != 0]['doc_id'][:pos_doc_num]
                # neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'][:3]
                neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'].sample(neg_doc_num)

                pos_doc_texts = [docstore.get(doc_id).text for doc_id in pos_doc_ids]
                neg_doc_texts = [docstore.get(doc_id).text for doc_id in neg_doc_ids]
            
            # if len(pos_doc_texts) == 0:
            #     continue

            jsonl_data.append({
                "query_id": query_id,
                "q_item_qid": q_item_qid,
                "query": query_text,
                "q_item_info": q_item_info,
                "adj_item_info": adj_item_info,
                "pos_doc": pos_doc_texts,
                "neg_doc": neg_doc_texts,
            })

        elif dataset_type == "test":

            jsonl_data.append({
                "query_id": query_id,
                "q_item_qid": q_item_qid,
                "query": query_text,
                "q_item_info": q_item_info,
                "adj_item_info": adj_item_info,
            })

        else:
            raise ValueError("dataset_type 值缺失/错误: “train”或者“test”")           

    # 将数据写入JSONL文件
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(jsonl_data)

    print("--------------------------------------------------------------")
    print(f"数据集构建完成 存储在了{output_file} 数据量为 {len(jsonl_data)}")
    print("--------------------------------------------------------------")


def build_dataset(
    docstore: NamedTuple,
    query_qid_file: str,
    qrels_file: str,
    item_info_file: str,
    adj_item_info_file: str,
    triple_id_file: str,
    output_file: str,
    adj_item_num: int=3,
    dataset_type: str="train",
    pos_doc_num: int=1,
    neg_doc_num: int=1
):
    try:
        query_qid_df = pd.read_csv(query_qid_file, encoding='utf-8').astype(str)
        item_info_df = pd.read_csv(item_info_file, encoding='utf-8').astype(str)
        adj_item_info_df = pd.read_csv(adj_item_info_file, encoding='utf-8').astype(str)
        triple_id_df = pd.read_csv(triple_id_file, encoding='utf-8').astype(str)
        qrels_df = pd.read_csv(qrels_file, encoding='utf-8')
        qrels_df['query_id'] = qrels_df['query_id'].astype(str)
        qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
        qrels_df['relevance'] = qrels_df['relevance'].astype(int)
    except Exception as e:
        logging.error(f"Error reading files: {e}")
        return

    jsonl_data = []

    for _, row in tqdm(query_qid_df.iterrows(), total=query_qid_df.shape[0], desc="Building dataset"):
        query_id = row['query_id']
        query_text = row['query_text']
        q_item_qid = row['q_item_qid']

        # Get the entity information corresponding to the query
        q_item = item_info_df.get(item_info_df['item_qid'] == q_item_qid)
        if q_item.empty:
            logging.warning(f"No item info found for q_item_qid: {q_item_qid}")
            continue

        # Check if any of the required fields are missing
        if q_item[['label_zh', 'label_kk', 'description_zh', 'description_kk']].isnull().any(axis=1).any():
            logging.warning(f"Missing required fields for q_item_qid: {q_item_qid}")
            continue

        q_item_info = {
            "label_zh": q_item['label_zh'].values[0],
            "label_kk": q_item['label_kk'].values[0],
            "description_zh": q_item['description_zh'].values[0],
            "description_kk": q_item['description_kk'].values[0]
        }

        adj_item_qids = triple_id_df[triple_id_df['item_qid'] == q_item_qid]['adj_item_qid']
        if adj_item_qids.empty:
            logging.warning(f"No adjacent items found for q_item_qid: {q_item_qid}")
            continue

        # Ensure we have the correct number of adjacent items
        adj_item_qids = adj_item_qids.sample(n=adj_item_num, replace=True) if len(adj_item_qids) < adj_item_num else adj_item_qids.sample(n=adj_item_num)

        adj_item_info = {
            "label_zh": [],
            "label_kk": [],
            "description_zh": [],
            "description_kk": []
        }

        stop_inner = False

        for adj_item_qid in adj_item_qids:
            adj_item = adj_item_info_df.get(adj_item_info_df['item_qid'] == adj_item_qid)

            # 其实这里可以不判断的，因为 triplet id 数据 已经经过 adj_item_info 过滤过了
            if adj_item.empty:
                logging.warning(f"No item info found for adj_item_qid: {adj_item_qid}")
                stop_inner = True
                break

            if adj_item[['label_zh', 'label_kk', 'description_zh', 'description_kk']].isnull().any(axis=1).any():
                logging.warning(f"Missing required fields for adj_item_qid: {adj_item_qid}")
                stop_inner = True
                break

            adj_item_info["label_zh"].append(adj_item['label_zh'].values[0])
            adj_item_info["label_kk"].append(adj_item['label_kk'].values[0])
            adj_item_info["description_zh"].append(adj_item['description_zh'].values[0])
            adj_item_info["description_kk"].append(adj_item['description_kk'].values[0])

        if stop_inner:
            continue

        if dataset_type == "train":
            query_docs = qrels_df[qrels_df['query_id'] == query_id]
            if query_docs.empty:
                logging.warning(f"No relevant documents found for query_id: {query_id}")
                continue

            pos_doc_ids = query_docs[query_docs['relevance'] != 0]['doc_id'].head(pos_doc_num)
            neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'].sample(n=neg_doc_num)

            pos_doc_texts = [docstore.get(doc_id).text for doc_id in pos_doc_ids]
            neg_doc_texts = [docstore.get(doc_id).text for doc_id in neg_doc_ids]

            jsonl_data.append({
                "query_id": query_id,
                "q_item_qid": q_item_qid,
                "query": query_text,
                "q_item_info": q_item_info,
                "adj_item_info": adj_item_info,
                "pos_doc": pos_doc_texts,
                "neg_doc": neg_doc_texts,
            })
        elif dataset_type == "test":
            jsonl_data.append({
                "query_id": query_id,
                "q_item_qid": q_item_qid,
                "query": query_text,
                "q_item_info": q_item_info,
                "adj_item_info": adj_item_info,
            })
        else:
            raise ValueError("dataset_type must be either 'train' or 'test'.")

    # Write data to JSONL file
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(jsonl_data)

    logging.info(f"Dataset built and saved to {output_file}. Data size: {len(jsonl_data)}")


if __name__ == "__main__":

    HOME_DIR = Path(__file__).parent.parent / 'base_data'

    # 加载 zh-kk clir 数据集
    CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
    # 加载原始查询数据
    # queries_df = pd.DataFrame(CLIRMatrix_dataset.queries_iter())
    # 加载 doc 数据
    docstore = CLIRMatrix_dataset.docs_store()
    # 加载原始的 qrels 数据
    qrels_df = pd.DataFrame(CLIRMatrix_dataset.qrels_iter())

    # -------------------- 构建 新的 qrels 文件 --------------------
    # 加载过滤 qrels 数据所需的参考文件
    # triple_id_file = str(HOME_DIR / 'base_train_triplet_id_fragment_5.csv')

    # qrels_filtered_df = filter_qrels(qrels_df=qrels_df, filter_reference_file=triple_id_file)
    # 过滤后 query 数量为 5092
    # 段落数量 201707

    # new_qrels_file = str(HOME_DIR / 'base_train_qrels.csv')
    # new_qrels_df = build_new_base_train_qrels(qrels_filtered_df, new_qrels_file, save_new_qrels=False)
    # print(new_qrels_df)
    # -------------------- 构建 新的 qrels 文件 --------------------


    build_dataset(
        docstore=docstore,
        query_qid_file=str(HOME_DIR / 'base_train_query_entity_qid_final.csv'),
        qrels_file=str(HOME_DIR / 'base_train_qrels.csv'),
        item_info_file=str(HOME_DIR / 'base_train_query_entity_info_filled.csv'),
        adj_item_info_file=str(HOME_DIR / 'base_train_adj_item_info_filled.csv'),
        triple_id_file=str(HOME_DIR / 'base_train_triplet_id_fragment_3_final.csv'),
        output_file=str(HOME_DIR / 'train_dataset.jsonl'),
        adj_item_num=3,
        dataset_type="train",
        pos_doc_num=1,
        neg_doc_num=1
    )

