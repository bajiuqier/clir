import openai
import pandas as pd

import random
import time
import re
import os


def random_id(seed: str) -> str:
    """
    将文本作为随机种子
    由于在xlsx文件中不能以0开头 故在前面添加“1”
    :param seed: 随机种子 使用query文本作为种子
    :return: 返回对应生成的数字字符串
    """
    random.seed(seed)
    return "1" + str(random.randint(0, 99999)).zfill(5)

def query_reorganize(lines: list) -> list:

    if len(lines) == 0:
        return ["0", "0"]

    queries_reorganized = []

    split_pattern = r'\.|\)|-|:'

    for item in lines:
        text = re.split(split_pattern, item, maxsplit=1)
        # text = item.split(".", 1)
        if len(text) > 1:
            if text[1].strip() != "":
                queries_reorganized.append(text[1].strip())
        else:
            if text[0].strip() != "":
                queries_reorganized.append(item.strip())

    if len(queries_reorganized) >= 2:
        return queries_reorganized
    elif len(queries_reorganized) == 1:
        queries_reorganized.append("0")
        return queries_reorganized
    else:
        return ["0", "0"]

if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    file_path = os.path.join(data_path, "passages_filter_1.xlsx")
    
    data_original = pd.read_excel(file_path)
    if data_original.isna().any().any():
        data_original.dropna(inplace=True)
        data_original.reset_index(inplace=True, drop=True)

    # key 1
    # openai.api_key = "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"
    # key 2
    # openai.api_key = "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"
    # 曹的
    openai.api_key = "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"

    Prompt = """\n请将上面的哈萨克语段落,生成两个用于问信息检索任务的query,并以列表的形式返回"""

    data_original[['query_1_id', 'query_1_kk', 'query_2_id', 'query_2_kk']] = None

    batch_size = 200
    start_index = 232
    global err
    err = None
    
    while start_index < len(data_original):
        if err :
            break
        batch = data_original[start_index:start_index + batch_size]
        print("Processing batch with {} rows".format(len(batch)))

        for index, row in batch.iterrows():
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        # {"role": "system", "content": Prompt},
                        {"role": "user", "content": row.passage_text + Prompt}
                    ]
                )
            except Exception as e:
                print(e)
                err = e
                print(f'start_index: {start_index}, index: {index}')
                err_pre_file_path = os.path.join(data_path, f'{start_index}_{index}.xlsx')
                batch.to_excel(err_pre_file_path, index=False)
                break
            else:

                unicode_string = response.choices[0].message.content
                
                # 如果返回的内容为空  直接进行下一个
                # if unicode_string.isspace():
                    # continue

                lines = unicode_string.split('\n')
                queries = query_reorganize(lines)
                query_1_kk = queries[0]
                query_2_kk = queries[1]

                batch.loc[index, "query_1_id"] = random_id(query_1_kk)
                batch.loc[index, "query_1_kk"] = query_1_kk
                batch.loc[index, "query_2_id"] = random_id(query_2_kk)
                batch.loc[index, "query_2_kk"] = query_2_kk
                
                time.sleep(30)

                if (index + 1) % 50 == 0:
                    print(f'The {index}th data has been processed')

        data_original_file_path = os.path.join(data_path, f"data_original_{start_index}.xlsx")
        batch.to_excel(data_original_file_path, index=False)
        print(f"The {start_index}th data query generation completed")

        # data_translate_file_path = os.path.join(data_path, f"data_translate_{start_index}.xlsx")
        # data_translate = query_translate(batch)
        # data_translate.to_excel(data_translate_file_path, index=False)
        # print(f"The {start_index}th data query translate completed")

        start_index += batch_size

