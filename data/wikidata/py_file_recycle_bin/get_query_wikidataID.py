import requests
import pandas as pd
import ir_datasets
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

HOME_DIR = Path(__file__).parent / 'data_file'
test1_QID_search_results_file = str(HOME_DIR / 'full_train_QID_search_results1.csv')

test1_dataset_obj = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train')
test1_queries_df = pd.DataFrame(test1_dataset_obj.queries_iter())
test1_queries_list = test1_queries_df['text'].to_list()

def fetch_data(query):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&format=json&errorformat=plaintext&language=zh&uselang=zh&type=item"
    response = requests.get(url)

    if response.status_code == 200:
        search_results = response.json()
    else:
        data_item = {'query': query, 'search_term': 'network error'}
        return data_item

    search_term = search_results.get('searchinfo', {}).get('search')
    data_item = {'query': query, 'search_term': search_term}

    if search_results.get('search') is not None:
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
    
    return data_item

def process_query(query):
    result = fetch_data(query)
    return result

if __name__ == "__main__":
    results = []
    # with tqdm(total=len(test1_queries_list)) as pbar:
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_query = {executor.submit(process_query, query): query for query in test1_queries_list[0:100000]}
        for future in tqdm(as_completed(future_to_query), total=len(test1_queries_list[0:100000])):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv(test1_QID_search_results_file, index=False, encoding='utf-8')
    # print(df)