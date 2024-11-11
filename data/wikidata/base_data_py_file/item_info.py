import requests
import json
from typing import Tuple, List, Union
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class WikidataItemInfoFetcher:
    """
    通过多线程的方式获取 Wikidata 条目信息
    """

    def __init__(self, processed_qids_file: str):
        self.processed_qids_file = processed_qids_file

    @staticmethod
    def fetch_item_info(qid: str) -> dict:
        """
        获取指定条目ID(item_qid)的信息

        Args:
            qid (str): 条目ID

        Returns:
            dict: 包含条目信息的字典
        """
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        response = requests.get(url)

        if response.status_code == 200:
            search_results = response.json()
        else:
            return {'qid': qid}

        labels = search_results['entities'].get(qid, {}).get('labels', {})
        descriptions = search_results['entities'].get(qid, {}).get('descriptions', {})

        label_zh = labels.get('zh', {}).get('value')
        label_kk = labels.get('kk', {}).get('value')
        label_en = labels.get('en', {}).get('value')

        description_zh = descriptions.get('zh', {}).get('value')
        description_kk = descriptions.get('kk', {}).get('value')
        description_en = descriptions.get('en', {}).get('value')

        data_item = {
            'qid': qid,
            'label_zh': label_zh,
            'label_kk': label_kk,
            'label_en': label_en,
            'description_zh': description_zh,
            'description_kk': description_kk,
            'description_en': description_en
        }

        return data_item

    def load_processed_qids(self) -> List[str]:
        """
        加载已处理的条目ID列表

        Returns:
            List[str]: 已处理的条目ID列表
        """
        try:
            with open(self.processed_qids_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed_qid = data.get("processed_qids", [])
            return processed_qid
        except (json.JSONDecodeError, FileNotFoundError):
            # 如果文件为空或无法解析JSON，则返回空列表
            return []
        except Exception as e:
            # 捕获其他可能的异常，并打印错误信息
            print(f"Error loading processed QIDs: {e}")
            return []

    def save_processed_qids(self, processed_qid: List[str]):
        """
        保存已处理的条目ID列表
        """
        with open(self.processed_qids_file, 'w') as f:
            data = {"processed_qids": processed_qid}
            json.dump(data, f, ensure_ascii=False, indent=4)

    def multithreading_fetch_item_info(self, qid_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        使用多线程的方式批量获取条目信息

        Args:
            qid_list (List[str]): 条目ID列表

        Returns:
            Tuple[pd.DataFrame, List[str]]: 包含条目信息的DataFrame和已处理的条目ID列表
        """
        results = []
        error_occurred = False
        lock = threading.Lock()
        processed_qids = self.load_processed_qids()

        qid_list = [qid for qid in qid_list if qid not in processed_qids]

        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_qid = {executor.submit(self.fetch_item_info, item_qid): item_qid for item_qid in qid_list}
            for future in tqdm(as_completed(future_to_qid), total=len(qid_list)):
                with lock:
                    if error_occurred:
                        break

                item_qid = future_to_qid[future]

                try:
                    result = future.result()
                    results.append(result)
                    with lock:
                        if isinstance(item_qid, str):
                            processed_qids.append(item_qid)
                        if isinstance(item_qid, list):
                            processed_qids.extend(item_qid)

                except Exception as e:
                    with lock:
                        error_occurred = True
                        print(f"Error occurred: {e}")
                        for f in future_to_qid:
                            f.cancel()
                        break

        df = pd.DataFrame(results)
        return df, processed_qids


# 示例usage
if __name__ == "__main__":
    HOME_DIR = Path(__file__).parent.parent / 'base_data_file'

    triplet_id_file = str(HOME_DIR / 'base_train_triplet_id_filtered.csv')
    # 假设这是您的工作目录中的一个文件，用于存储已处理的QID
    PROCESSED_QIDS_FILE = str(HOME_DIR / "processed_qids.json")

    # 在尝试加载之前，检查文件是否存在，如果不存在，则创建它
    if not Path(PROCESSED_QIDS_FILE).exists():
        # 创建一个新的空文件
        Path(PROCESSED_QIDS_FILE).touch()

    fetcher = WikidataItemInfoFetcher(processed_qids_file=PROCESSED_QIDS_FILE)

    qid_df = pd.read_csv(triplet_id_file, encoding='utf-8').astype(str)
    qid_list = qid_df.loc[:]['adj_item_qid'].to_list()
    # 去除列表中的重复值
    qid_list = list(set(qid_list))

    processed_qids = fetcher.load_processed_qids()

    loop_num = 0

    while len(processed_qids) < len(qid_list):
        print(f"---------------- 第 {loop_num} 次处理数据 ----------------")

        property_info_df, processed_qids = fetcher.multithreading_fetch_item_info(qid_list)

        adj_item_info_file = str(HOME_DIR / f"base_train_adj_item_info_{loop_num}.csv")

        property_info_df.to_csv(adj_item_info_file, index=False, encoding='utf-8')

        loop_num = loop_num + 1

        fetcher.save_processed_qids(processed_qid=processed_qids)

        if loop_num == 20:
            break

    if len(processed_qids) < len(qid_list):
        print("-------------------------------------------------------")
        print(f"数据还没有处理完成！ 文件存储在了{HOME_DIR}")
        print("-------------------------------------------------------")
    else:
        print("-------------------------------------------------------")
        print(f"数据处理完成！ 文件存储在了{HOME_DIR}")
        print("-------------------------------------------------------")
