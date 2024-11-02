import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

HOME_DIR = Path(__file__).parent.parent / 'base_data'

def fetch_triplet_id(query_entity_qid: str) -> list:
    # 查询语句，使用格式化字符串将 entity_id 插入到查询中
    query = f"""
    SELECT ?item ?p ?adjItem
    WHERE {{
        VALUES (?item) {{(wd:{query_entity_qid})}}
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

    triplet_id = []
    if isinstance(search_results, int):
        triplet_id_item = {'item_qid': query_entity_qid}
        triplet_id.append(triplet_id_item)
        return triplet_id

    else:
        for result in search_results['results']['bindings']:
            item_qid = result['item']['value'].split('/')[-1]
            prop_qid = result['p']['value'].split('/')[-1]
            adj_item_qid = result['adjItem']['value'].split('/')[-1]
            
            if item_qid == query_entity_qid:
                # 新建字典 
                # 同一个字典 在内循环中 如果修改的话，是在原来的 字典上进行修改的 这样就会将之前存在 triplet_data 的数据覆盖
                triplet_id_item = {'item_qid': query_entity_qid, 'property_qid': prop_qid, 'adj_item_qid': adj_item_qid}
                triplet_id.append(triplet_id_item)
        return triplet_id


if __name__ == "__main__":
    query_entity_info_file = str(HOME_DIR / 'base_test1_query_entity_info_filtered.csv')

    triplet_id_file = str(HOME_DIR / 'base_test11_triplet_id.csv')

    query_entity_info_df = pd.read_csv(query_entity_info_file, encoding='utf-8').astype(str)

    triplet_id = []

    start_idx = 0
    end_idx = query_entity_info_df.shape[0]
    pending_query_qid_df = query_entity_info_df.iloc[start_idx:end_idx]

    for idx, row in tqdm(pending_query_qid_df.iterrows(), total=pending_query_qid_df.shape[0]):
        
        query_entity_qid = row['item_qid']
        
        try:
            result = fetch_triplet_id(query_entity_qid=query_entity_qid)
        except Exception as e:
            print(f"------------ 处理到第{idx}个数据时 报错了 ------------")
            print(f"Error occurred: {e}")
            triplet_id_file = str(HOME_DIR / f'base_test11_triplet_id_{idx}.csv')

            break

        triplet_id.append(result)

    triplet_id = sum(triplet_id, [])

    triplet_id_df = pd.DataFrame(triplet_id)
    triplet_id_df.to_csv(triplet_id_file, index=False, encoding='utf-8')
