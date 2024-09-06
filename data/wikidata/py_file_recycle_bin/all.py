import requests
import pandas as pd
import ir_datasets
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from get_query_wikidataID import process_query
from get_item_info import get_item_info
from data_processing import filter_qid

HOME_DIR = Path(__file__).parent / 'data_file_1'

full_dataset_obj = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train')
queries_df = pd.DataFrame(full_dataset_obj.queries_iter())
queries_list = queries_df['text'].to_list()

batch_size = 50000

for index in range(500000, len(queries_list), batch_size):
    QID_search_results_file = str(HOME_DIR / f'full_train_QID_search_results{index}.csv')

    QID_search_results = []
    # with tqdm(total=len(test1_queries_list)) as pbar:
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_query = {executor.submit(process_query, query): query for query in queries_list[index:index + batch_size]}
        for future in tqdm(as_completed(future_to_query), total=len(queries_list[index:index + batch_size])):
            QID_search_results.append(future.result())

    QID_df = pd.DataFrame(QID_search_results)
    QID_df.to_csv(QID_search_results_file, index=False, encoding='utf-8')

    QID_search_results_filtered_file = str(HOME_DIR / f'full_train_QID_filtered_search_results{index}.csv')
    filter_qid(QID_search_results_file, QID_search_results_filtered_file)

    query_entity_info_file = str(HOME_DIR / f'full_train_query_entity_info{index}.csv')
    QID_filtered_df = pd.read_csv(QID_search_results_filtered_file, encoding='utf-8')
    QID_list = QID_filtered_df.loc[:]['id'].to_list()

    query_entity_info_data = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_query = {executor.submit(get_item_info, item_id): item_id for item_id in QID_list}
        for future in tqdm(as_completed(future_to_query), total=len(QID_list)):
            query_entity_info_data.append(future.result())

    query_entity_info_df = pd.DataFrame(query_entity_info_data)
    query_entity_info_df.to_csv(query_entity_info_file, index=False, encoding='utf-8')

    query_entity_info_kk_file = str(HOME_DIR / f'full_train_query_entity_info_kk{index}.csv')
    # 删除 label_kk description_kk 为空的数据
    query_entity_info_df = query_entity_info_df.dropna(subset=['label_kk', 'description_kk'], how='any')
    query_entity_info_df.to_csv(query_entity_info_kk_file, index=False, encoding='utf-8')