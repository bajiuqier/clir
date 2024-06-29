import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

HOME_DIR = Path(__file__).parent / 'data_file_2'
QID_file = str(HOME_DIR / 'merged_kk_item_info.csv')
triplet_id_search_results_file = str(HOME_DIR / 'triplet_id_search_results.csv')

def get_triplet_id(entity_id):
    # 查询语句，使用格式化字符串将 entity_id 插入到查询中
    query = f"""
    SELECT ?item ?p ?adjItem
    WHERE {{
        VALUES (?item) {{(wd:{entity_id})}}
        {{
            ?item ?p ?adjItem .
            FILTER(isIRI(?adjItem))
            FILTER(STRSTARTS(STR(?p), 'http://www.wikidata.org/prop/direct/'))
        }} UNION {{
            ?adjItem ?p ?item .
            FILTER(isIRI(?adjItem))
            FILTER(STRSTARTS(STR(?p), 'http://www.wikidata.org/prop/direct/'))
        }}
    }}
    """

    url = 'https://query.wikidata.org/sparql'
    params = {'query': query, 'format': 'json'}

    # 发送请求
    response = requests.get(url, params=params)

    # 检查请求是否成功
    if response.status_code == 200:
        search_results = response.json()
    else:
        # raise Exception(f"Query failed with status code {response.status_code}")
        search_results = response.status_code

    if isinstance(search_results, int):
        triplet_data_item = {'item': entity_id}
        return triplet_data_item
        # triplet_data.append(triplet_data_item)
    else:
        for result in search_results['results']['bindings']:
            item = result['item']['value'].split('/')[-1]
            prop = result['p']['value'].split('/')[-1]
            adjItem = result['adjItem']['value'].split('/')[-1]
            # 新建字典 
            # 同一个字典 在内循环中 如果修改的话，是在原来的 字典上进行修改的 这样就会将之前存在 triplet_data 的数据覆盖
            triplet_data_item = {'original_item': entity_id, 'item': item, 'property': prop, 'adjItem': adjItem}
            return triplet_data_item


QID_df = pd.read_csv(QID_file, encoding='utf-8')
QID_list = QID_df.loc[:]['item'].to_list()

triplet_data = []

with ThreadPoolExecutor(max_workers=16) as executor:
    future_to_item = {executor.submit(get_triplet_id, item_id): item_id for item_id in QID_list}
    for future in tqdm(as_completed(future_to_item), total=len(QID_list)):
        triplet_data.append(future.result())

# 存储数据
triplet_id_df = pd.DataFrame(triplet_data)
triplet_id_df.to_csv(triplet_id_search_results_file, index=False, encoding='utf-8')
