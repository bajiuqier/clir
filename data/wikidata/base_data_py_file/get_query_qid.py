import requests
import pandas as pd
import ir_datasets
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

HOME_DIR = Path(__file__).parent.parent / 'base_data'
# 定义 查询 对应 qid 的文件
query_entity_qid_file = str(HOME_DIR / 'base_test2_query_entity_qid.csv')

# 加载 zh-kk base-train 查询数据 并将其转成列表
dataset_obj = ir_datasets.load('clirmatrix/kk/bi139-base/zh/test2')
queries_df = pd.DataFrame(dataset_obj.queries_iter())


# queries_list = queries_df['text'].to_list()

def fetch_qid(query_id, query_text):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query_text}&format=json&errorformat=plaintext&language=zh&uselang=zh&type=item"
    response = requests.get(url)

    # 访问成功 获取其 json 数据
    if response.status_code == 200:
        search_results = response.json()
    # 否则 返回 网络错误 并且返回 query_id 和 query text 信息
    else:
        data_item = {'query_id': query_id, 'query_text': query_text, 'search_term': 'network error'}
        return data_item

    # search_term 在不出错的情况下 应该是和 query_text 相同的
    search_term = search_results.get('searchinfo', {}).get('search')
    data_item = {'query_id': query_id, 'query': query_text, 'search_term': search_term}

    # search_results.get('search') 得到的是一个列表 可能是一个空列表 []
    if len(search_results.get('search')) > 0:
        # 获取 检索结果的第一条数据
        item = search_results['search'][0]

        # 再判断一下 结果 的 label 值是否与 search_term 或者 query_text 相同
        if item['label'] == search_term:
            # if item['label'] == query_text:
            data_item['q_item_qid'] = item.get('id')
            data_item['label'] = item.get('label')
            data_item['description'] = item.get('description')

    return data_item


def process_query(query_id, query_text):
    result = fetch_qid(query_id, query_text)
    return result


if __name__ == "__main__":
    results = []
    error_occurred = False  # 用于记录是否发生错误

    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_query = {}
        for _, row in queries_df.iterrows():
            future = executor.submit(process_query, row['query_id'], row['text'])
            future_to_query[future] = (row['query_id'], row['text'])

        for future in tqdm(as_completed(future_to_query), total=len(queries_df)):
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

    # for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
    #     results.append(process_query(row['query_id'], row['text']))

    if not error_occurred:
        df = pd.DataFrame(results)
        df.to_csv(query_entity_qid_file, index=False, encoding='utf-8')
