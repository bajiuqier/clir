import requests
import pandas as pd
import ir_datasets
from pathlib import Path
from tqdm import tqdm

HOME_DIR = Path(__file__).parent
test1_QID_search_results_file = str(HOME_DIR / 'test1_QID_search_results.csv')

test1_dataset_obj = ir_datasets.load('clirmatrix/kk/bi139-base/zh/test1')
test1_queries_df = pd.DataFrame(test1_dataset_obj.queries_iter())
test1_queries_list = test1_queries_df.loc[:]['text'].to_list()

data = []

for query in tqdm(test1_queries_list, total=len(test1_queries_list)):

    # url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=%E2%80%9C%E4%B8%89%E4%B8%AA%E4%BB%A3%E8%A1%A8%E2%80%9D%E9%87%8D%E8%A6%81%E6%80%9D%E6%83%B3&format=json&errorformat=plaintext&language=zh&uselang=zh&type=item"
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&format=json&errorformat=plaintext&language=zh&uselang=zh&type=item"

    response = requests.get(url)
    search_results = response.json()
    # data 不会为空 匹配不到 search 的内容会为空
    # {'searchinfo': {'search': 'xxxxxxxxx'}, 'search': [], 'success': 1}

    # 搜索关键词
    search_term = search_results['searchinfo'].get('search')

    # 将 query 和 检索的内容同时存储，用户后面检查 是否存在偏差
    data_item = {'query': query, 'search_term': search_term}

    if len(search_results['search']) != 0:
        for item_idx, item in enumerate(search_results['search']):
            if item_idx == 0:
                data_item['id'] = item.get('id')
                data_item['label'] = item.get('label')
                data_item['description'] = item.get('description')
            else:
                if item['label'] == search_term:
                    data_item[f'id_{item_idx}'] = item.get('id')
                    data_item[f'label_{item_idx}'] = item.get('label')
                    data_item[f'description_{item_idx}'] = item.get('description')
            
    data.append(data_item)

# 将数据存储为 DataFrame 格式
df = pd.DataFrame(data)
df.to_csv(test1_QID_search_results_file, index=False, encoding='utf-8')
# 打印 DataFrame
print(df)