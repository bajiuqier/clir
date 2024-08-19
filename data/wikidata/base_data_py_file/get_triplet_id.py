import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

HOME_DIR = Path(__file__).parent.parent / 'base_data'

def fetch_triplet_id(query_id: str, query_entity_qid: str) -> list:
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
        triplet_id_item = {'query_id': query_id, 'item_qid': query_entity_qid}
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
                triplet_id_item = {'query_id': query_id, 'item_qid': query_entity_qid, 'property_qid': prop_qid, 'adj_item_qid': adj_item_qid}
                triplet_id.append(triplet_id_item)
        return triplet_id

# 定义一个方法 把 查询实体对应的qid 的数据 过滤一下
# 因为 query_entity_info 的数据 过滤后 是有所缺失的
# 去除 在query_entity_info中 没有信息的 query
def building_new_query_qid(query_qid_file: str, entity_info_file: str) -> pd.DataFrame:
    query_qid_df = pd.read_csv(query_qid_file, encoding='utf-8')
    entity_info_df = pd.read_csv(entity_info_file, encoding='utf-8')

    query_qid_df['qid'] = query_qid_df['qid'].astype(str)
    entity_info_df['item_qid'] = entity_info_df['item_qid'].astype(str)

    new_query_qid_df = pd.merge(query_qid_df, entity_info_df, left_on='qid', right_on='item_qid', how='left')
    # 删除 item_qid 为空的行数据
    new_query_qid_df.dropna(subset=['item_qid'], inplace=True)
    new_query_qid_df = new_query_qid_df[['query_id', 'query', 'qid']]

    return new_query_qid_df


query_qid_file = str(HOME_DIR / 'base_test2_QID_filtered_search_results.csv')
entity_info_file = str(HOME_DIR / 'base_test2_query_entity_filtered_info.csv')

# new_qurey_qid_file = str(HOME_DIR / 'new_qurey_qid_get_triple_id.csv')

triplet_id_file = str(HOME_DIR / 'base_test2_triplet_id.csv')

# 构建新的 query_qid
query_qid_df = building_new_query_qid(query_qid_file, entity_info_file)
# query_qid_df.to_csv(new_qurey_qid_file, index=False, encoding='utf-8')

# query_qid_df = pd.read_csv(new_qurey_qid_file, encoding='utf-8')
query_qid_df = query_qid_df.astype(str)

triplet_id = []

start_idx = 595
end_idx = query_qid_df.shape[0]
pending_query_qid_df = query_qid_df.iloc[start_idx:end_idx]

for idx, row in tqdm(pending_query_qid_df.iterrows(), total=pending_query_qid_df.shape[0]):
    query_id = row['query_id']
    query_entity_qid = row['qid']
    
    try:
        result = fetch_triplet_id(query_id=query_id, query_entity_qid=query_entity_qid)
    except Exception as e:
        print(f"------------ 处理到第{idx}个数据时 报错了 ------------")
        print(f"Error occurred: {e}")
        triplet_id_file = str(HOME_DIR / f'triplet_id_{idx}.csv')

        break

    triplet_id.append(result)

triplet_id = sum(triplet_id, [])

triplet_id_df = pd.DataFrame(triplet_id)
triplet_id_df.to_csv(triplet_id_file, index=False, encoding='utf-8')
