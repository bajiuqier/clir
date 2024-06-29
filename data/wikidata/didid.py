import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


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

    try:
        # 发送请求
        response = requests.get(url, params=params)


        # 检查请求是否成功
        if response.status_code == 200:
            return response.json()
        else:
            return response.status_code

        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for entity {entity_id}: {e}")
        return 66

QID_df = pd.read_csv(QID_file, encoding='utf-8')
QID_list = QID_df['item'].to_list()

triplet_data = []

for item_id in tqdm(QID_list, total=len(QID_list)):
    search_results = get_triplet_id(item_id)
    if isinstance(search_results, int):
        triplet_data_item = {'item': item_id}
        triplet_data.append(triplet_data_item)
    else:
        for result in search_results['results']['bindings']:
            item = result['item']['value'].split('/')[-1]
            prop = result['p']['value'].split('/')[-1]
            adjItem = result['adjItem']['value'].split('/')[-1]
            triplet_data_item = {'original_item': item_id, 'item': item, 'property': prop, 'adjItem': adjItem}
            triplet_data.append(triplet_data_item)


# 存储数据
triplet_id_df = pd.DataFrame(triplet_data)
triplet_id_df.to_csv(triplet_id_search_results_file, index=False, encoding='utf-8')
