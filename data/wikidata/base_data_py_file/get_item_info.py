import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_item_info(item_qid:str):

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{item_qid}.json"  
    response = requests.get(url)

    if response.status_code == 200:
        search_results = response.json()
    else:
        data_item = {'item_qid': item_qid}
        return data_item

    # language = ['zh', 'kk']
    labels = search_results['entities'].get(item_qid, {}).get('labels', {})
    descriptions = search_results['entities'].get(item_qid, {}).get('descriptions', {})

    # 获取 label 值
    # get('zh', {}) 安全地尝试获取 zh 键的值 如果 zh 不存在 则返回一个空字典
    # 这样 就算 get('zh') 为空 get('value') 也不会报错
    label_zh = labels.get('zh', {}).get('value')
    label_kk = labels.get('kk', {}).get('value')
    label_en = labels.get('en', {}).get('value')

    # 获取 description 值
    description_zh = descriptions.get('zh', {}).get('value')
    description_kk = descriptions.get('kk', {}).get('value')
    description_en = descriptions.get('en', {}).get('value')

    data_item = {
        'item_qid': item_qid,
        'label_zh': label_zh,
        'label_kk': label_kk,
        'label_en': label_en,
        'description_zh': description_zh,
        'description_kk': description_kk,
        'description_en': description_en,
    }

    return data_item


def process_item_qid(item_qid:str):
    result = fetch_item_info(item_qid=item_qid)
    return result

def multithreading_fetch_item_info(qid_list:list) -> pd.DataFrame:
    results = []
    error_occurred = False  # 用于记录是否发生错误

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_query = {executor.submit(process_item_qid, item_qid): item_qid for item_qid in qid_list}
        for future in tqdm(as_completed(future_to_query), total=len(qid_list)):
            if error_occurred:
                break  # 如果发生错误，立即停止处理其他任务

            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                error_occurred = True
                print(f"Error occurred: {e}")
                # 如果发生错误，立即取消所有剩余的任务
                for f in future_to_query:
                    f.cancel()
                break

    if not error_occurred:
        df = pd.DataFrame(results)
        return df

if __name__ == "__main__":

    HOME_DIR = Path(__file__).parent

    # 先获取 query 对应的实体的 info 因为有的 实体可能没有相关信息 就可以删掉 在接下来的 过程中可以减少一些数据
    query_entity_qid_file = str(HOME_DIR / 'base_train_QID_filtered_search_results.csv')
    query_entity_info_file = str(HOME_DIR / 'base_train_query_entity_info.csv')

    query_entity_qid_df = pd.read_csv(query_entity_qid_file, encoding='utf-8')
    query_entity_qid_df['qid'] = query_entity_qid_df['qid'].astype(str)
    query_entity_qid_list = query_entity_qid_df.loc[:]['qid'].to_list()

    if isinstance(query_entity_qid_list[0], str):
        query_entity_info_df = multithreading_fetch_item_info(qid_list=query_entity_qid_list)
        query_entity_info_df.to_csv(query_entity_info_file, index=False, encoding='utf-8')
    else:
        ValueError('qid_list 里面的 qid 需要是 str 类型')


