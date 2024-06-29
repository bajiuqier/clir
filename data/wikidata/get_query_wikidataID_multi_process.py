import requests
import pandas as pd
import ir_datasets
from pathlib import Path
from tqdm import tqdm
import multiprocessing

HOME_DIR = Path(__file__).parent
test1_QID_search_results_file = str(HOME_DIR / 'base_train_QID_search_results.csv')

test1_dataset_obj = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
test1_queries_df = pd.DataFrame(test1_dataset_obj.queries_iter())
test1_queries_list = test1_queries_df['text'].to_list()

def fetch_data(query):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&format=json&errorformat=plaintext&language=zh&uselang=zh&type=item"
    response = requests.get(url)
    search_results = response.json()

    search_term = search_results['searchinfo'].get('search')
    data_item = {'query': query, 'search_term': search_term}

    if search_results['search']:
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

def update_progress(result):
    global pbar
    pbar.update()

if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        with tqdm(total=len(test1_queries_list)) as pbar:
            results = []
            for query in test1_queries_list:
                result = pool.apply_async(fetch_data, args=(query,), callback=update_progress)
                results.append(result)
            
            pool.close()
            pool.join()

            results = [result.get() for result in results]

    df = pd.DataFrame(results)
    df.to_csv(test1_QID_search_results_file, index=False, encoding='utf-8')

    print(df)
