import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

HOME_DIR = Path(__file__).parent
test1_QID_filtered_search_results_file = str(HOME_DIR / 'test1_QID_filtered_search_results.csv')
test1_QID_filtered_df = pd.read_csv(test1_QID_filtered_search_results_file, encoding='utf-8')
test1_triplet_id_search_results_file = str(HOME_DIR / 'test1_triplet_id_search_results.csv')
QID = test1_QID_filtered_df.loc[:]['id'].to_list()

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
        return response.json()
    else:
        # raise Exception(f"Query failed with status code {response.status_code}")
        return response.status_code
    

triplet_data = []
for qid in tqdm(QID, total=len(QID)):

    triplet_data_item = {'item': qid}

    search_results = get_triplet_id(entity_id=qid)
    
    if isinstance(search_results, int):
        triplet_data.append(triplet_data_item)
    else:
        # 处理响应数据
        for result in search_results['results']['bindings']:
                item = result['item']['value'].split('/')[-1]
                prop = result['p']['value'].split('/')[-1]
                adjItem = result['adjItem']['value'].split('/')[-1]

                triplet_data_item['item'] = item
                triplet_data_item['property'] = prop
                triplet_data_item['adjItem'] = adjItem
                triplet_data.append(triplet_data_item)

# 存储数据
test1_triplet_id_df = pd.DataFrame(triplet_data)
test1_triplet_id_df.to_csv(test1_triplet_id_search_results_file, index=False, encoding='utf-8')




