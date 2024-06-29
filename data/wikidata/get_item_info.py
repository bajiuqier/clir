import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

HOME_DIR = Path(__file__).parent / 'data_file'

# triplet_id_filtered_search_results_file = str(HOME_DIR / 'full_train_QID_filtered_search_results1.csv')
full_train_QID_filtered_search_results_file = str(HOME_DIR / 'full_train_QID_filtered_search_results1.csv')
query_entity_info_file = str(HOME_DIR / 'full_train_query_entity_info1.csv')
# query_adjacent_entity_info_file = str(HOME_DIR / 'base_train_query_adjacent_entity_info.csv')
# property_info_file = str(HOME_DIR / 'base_train_property_info.csv')

def get_item_info(item_id):

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{item_id}.json"

    try:
        
        response = requests.get(url)

        # 检查请求是否成功
        if response.status_code == 200:
            search_results = response.json()
        else:
            # raise Exception(f"Query failed with status code {response.status_code}")
            search_results = response.status_code

        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for entity {item_id}: {e}")
        search_results = 66
    
    if isinstance(search_results, int):
        data_item = {'item': f'{item_id}'}
    else:
        # language = ['zh', 'kk']
        labels = search_results['entities'].get(f'{item_id}', {}).get('labels', {})
        descriptions = search_results['entities'].get(f'{item_id}', {}).get('descriptions', {})

        # get('zh', {}) 安全地尝试获取 zh 键的值 如果 zh 不存在 则返回一个空字典
        # 这样 就算 get('zh') 为空 get('value') 也不会报错
        label_zh = labels.get('zh', {}).get('value')
        label_kk = labels.get('kk', {}).get('value')
        label_en = labels.get('en', {}).get('value')

        description_zh = descriptions.get('zh', {}).get('value')
        description_kk = descriptions.get('kk', {}).get('value')
        description_en = descriptions.get('en', {}).get('value')

        data_item = {
            'item': f'{item_id}',
            'label_zh': label_zh,
            'label_kk': label_kk,
            'label_en': label_en,
            'description_zh': description_zh,
            'description_kk': description_kk,
            'description_en': description_en,
        }

    return data_item
    query_entity_info_data.append(data_item)

if __name__ == "__main__":
    
    # triplet_id_filtered_df = pd.read_csv(triplet_id_filtered_search_results_file, encoding='utf-8')
    QID_filtered_df = pd.read_csv(full_train_QID_filtered_search_results_file, encoding='utf-8')
    QID_list = QID_filtered_df.loc[:]['id'].to_list()
    # qid_group = triplet_id_filtered_df.groupby('item')
    # adj_qid_group = triplet_id_filtered_df.groupby('adjItem')
    # prop_group = triplet_id_filtered_df.groupby('property')

    query_entity_info_data = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_query = {executor.submit(get_item_info, item_id): item_id for item_id in QID_list}
        for future in tqdm(as_completed(future_to_query), total=len(QID_list)):
            query_entity_info_data.append(future.result())

    query_entity_info_df = pd.DataFrame(query_entity_info_data)
    query_entity_info_df.to_csv(query_entity_info_file, index=False, encoding='utf-8')

