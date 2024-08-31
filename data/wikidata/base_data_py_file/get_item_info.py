import requests
import json
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

# 数据量少时可以用这个方法获取 item info
# def multithreading_fetch_item_info(qid_list:list) -> pd.DataFrame:
#     results = []
#     error_occurred = False  # 用于记录是否发生错误

#     with ThreadPoolExecutor(max_workers=16) as executor:
#         future_to_qid = {executor.submit(process_item_qid, item_qid): item_qid for item_qid in qid_list}
#         for future in tqdm(as_completed(future_to_qid), total=len(qid_list)):
#             if error_occurred:
#                 break  # 如果发生错误，立即停止处理其他任务

#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception as e:
#                 error_occurred = True
#                 print(f"Error occurred: {e}")
#                 # 如果发生错误，立即取消所有剩余的任务
#                 for f in future_to_qid:
#                     f.cancel()
#                 break

#     if not error_occurred:
#         df = pd.DataFrame(results)
#         return df


# 加载 已经处理的 qid 数据
def load_processed_qids(PROCESSED_QIDS_FILE: str):
    try:
        # 尝试打开文件并加载JSON内容
        with open(PROCESSED_QIDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 如果数据中包含"processed_qids"键，则返回其值；否则返回空列表
            processed_qids = data.get("processed_qids", [])
            return processed_qids
    except (json.JSONDecodeError, FileNotFoundError):
        # 如果文件为空或无法解析JSON，则返回空列表
        return []
    except Exception as e:
        # 捕获其他可能的异常，并打印错误信息
        print(f"Error loading processed QIDs: {e}")
        return []


# 保存已处理的QID
def save_processed_qids(processed_qids: list, PROCESSED_QIDS_FILE: str):
    with open(PROCESSED_QIDS_FILE, 'w') as f:
        data = {"processed_qids": processed_qids}
        json.dump(data, f, ensure_ascii=False, indent=4)


def multithreading_fetch_item_info(qid_list:list, processed_qids: list) -> pd.DataFrame:
    results = []
    error_occurred = False  # 用于记录是否发生错误

    qid_list = [qid for qid in qid_list if qid not in processed_qids]

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_qid = {executor.submit(process_item_qid, item_qid): item_qid for item_qid in qid_list}
        for future in tqdm(as_completed(future_to_qid), total=len(qid_list)):

            if error_occurred:
                break  # 如果发生错误，立即停止处理其他任务
            
            item_qid = future_to_qid[future]

            try:
                result = future.result()
                results.append(result)

                if isinstance(item_qid, str):
                    processed_qids.append(item_qid)
                if isinstance(item_qid, list):
                    processed_qids.extend(item_qid)

            except Exception as e:
                error_occurred = True
                print(f"Error occurred: {e}")
                # 如果发生错误，立即取消所有剩余的任务
                for f in future_to_qid:
                    f.cancel()
                break

    df = pd.DataFrame(results)
    return df, processed_qids


if __name__ == "__main__":

    HOME_DIR = Path(__file__).parent.parent / 'base_data'

    # ------------------ 获取 query 对应的实体的 info ------------------
    # 先获取 query 对应的实体的 info 因为有的 实体可能没有相关信息 就可以删掉 在接下来的 过程中可以减少一些数据
    # query_entity_qid_file = str(HOME_DIR / 'base_test2_QID_filtered_search_results.csv')
    # query_entity_info_file = str(HOME_DIR / 'base_test2_query_entity_info.csv')

    # query_entity_qid_df = pd.read_csv(query_entity_qid_file, encoding='utf-8')
    # query_entity_qid_df['qid'] = query_entity_qid_df['qid'].astype(str)
    # query_entity_qid_list = query_entity_qid_df.loc[:]['qid'].to_list()

    # if isinstance(query_entity_qid_list[0], str):
    #     query_entity_info_df = multithreading_fetch_item_info(qid_list=query_entity_qid_list)
    #     query_entity_info_df.to_csv(query_entity_info_file, index=False, encoding='utf-8')
    # else:
    #     raise ValueError('qid_list 里面的 qid 需要是 str 类型')

    
    # # ------------------ 获取属性信息 property_info ------------------
    # triplet_id_file = str(HOME_DIR / 'triplet_id_filtered.csv')
    # property_info_file = str(HOME_DIR / 'property_info.csv')

    # triplet_id_df = pd.read_csv(triplet_id_file, encoding='utf-8')
    # triplet_id_df = triplet_id_df.astype(str)

    # property_qid_list = triplet_id_df.loc[:]['property_qid'].to_list()
    # # 去除 列表中的重复值
    # property_qid_list = list(set(property_qid_list))

    # if isinstance(property_qid_list[0], str):
    #     property_info_df = multithreading_fetch_item_info(qid_list=property_qid_list)
    #     property_info_df.to_csv(property_info_file, index=False, encoding='utf-8')

    # else:
    #     raise ValueError('qid_list 里面的 qid 需要是 str 类型')

    # ------------------ 获取相邻实体的 adj_item_info ------------------

    # ADJ_ITEM_INFO_HOME_DIR = Path(__file__).parent.parent / 'base_train_adj_item_info'

    # triplet_id_file = str(HOME_DIR / 'base_train_triplet_id_fragment_3.csv')

    # triplet_id_df = pd.read_csv(triplet_id_file, encoding='utf-8')
    # triplet_id_df = triplet_id_df.astype(str)

    # adj_item_qid_list = triplet_id_df.loc[:]['adj_item_qid'].to_list()
    # # 去除 列表中的重复值
    # adj_item_qid_list = list(set(adj_item_qid_list))

    # # 假设这是您的工作目录中的一个文件，用于存储已处理的QID
    # PROCESSED_QIDS_FILE = str(ADJ_ITEM_INFO_HOME_DIR / "processed_qids.json")

    # loop_num = 0

    # if isinstance(adj_item_qid_list[0], str):

    #     # 读取已经 处理的 item qid
    #     processed_qids = load_processed_qids(PROCESSED_QIDS_FILE=PROCESSED_QIDS_FILE)  # 获取已经处理过的 item 的 qid

    #     while len(processed_qids) < len(adj_item_qid_list):

    #         print("-------------------------------------------------------")
    #         print(f"第 {loop_num} 次处理数据")
    #         print("-------------------------------------------------------")
            
    #         property_info_df, processed_qids = multithreading_fetch_item_info(qid_list=adj_item_qid_list, processed_qids=processed_qids)

    #         adj_item_info_file = str(ADJ_ITEM_INFO_HOME_DIR / f"base_train_adj_item_info_{loop_num}.csv")

    #         property_info_df.to_csv(adj_item_info_file, index=False, encoding='utf-8')

    #         loop_num = loop_num + 1

    #         save_processed_qids(processed_qids=processed_qids, PROCESSED_QIDS_FILE=PROCESSED_QIDS_FILE)

    # else:
    #     raise ValueError('qid_list 里面的 qid 需要是 str 类型')
    
    # print("-------------------------------------------------------")
    # print(f"数据处理完成 一共循环了{loop_num}次 文件存储在了{ADJ_ITEM_INFO_HOME_DIR}")
    # print("-------------------------------------------------------")
