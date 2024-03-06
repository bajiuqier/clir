import openai
import pandas as pd
from googletrans import Translator
import random
import time
import re


def random_id(seed: str) -> str:
    """
    将文本作为随机种子
    由于在xlsx文件中不能以0开头 故在前面添加“1”
    :param seed: 随机种子 使用query文本作为种子
    :return: 返回对应生成的数字字符串
    """
    random.seed(seed)
    return "1" + str(random.randint(0, 999999)).zfill(6)

# 设置Google翻译服务地址
translator = Translator(service_urls=['translate.google.com'])
def translate(sentences: str) -> str:
    # result = translator.translate(sentences, src='kk', dest='zh-cn')  #dest:目标语言
    result = translator.translate(sentences, dest='zh-cn').text  #dest:目标语言
    return result
  
def query_translate(df: pd.DataFrame) -> pd.DataFrame:

    df[["query_1_zh", "query_2_zh"]] = None

    for index, row in df.iterrows():
        if not pd.isna(row.query_1):
            df.loc[index, "query_1_zh"] = translate(row.query_1)
        if not pd.isna(row.query_2):
            df.loc[index, "query_2_zh"] = translate(row.query_2)
    
    return df

def query_reorganize(lines: list) -> list:

    if len(lines) == 0:
        return ["0", "0"]

    queries_reorganized = []

    split_pattern = r'.|\)|-'

    for item in lines:
        text = re.split(split_pattern, item, maxsplit=1)
        # text = item.split(".", 1)
        if len(text) > 1:
            if text[1].strip() != "":
                queries_reorganized.append(text[1].strip())
            else:
                continue
        else:
            queries_reorganized.append(item.strip())

    if len(queries_reorganized) >= 2:
        return queries_reorganized
    elif len(queries_reorganized) == 1:
        queries_reorganized.append("0")
        return queries_reorganized
    else:
        return ["0", "0"]

if __name__ == "__main__":

    passages_df = pd.read_excel("./data_collection.xlsx")

    openai.api_key = "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"

    Prompt = """请将上面的哈萨克语段落,生成两个用于问信息检索任务的query,并以列表的形式返回"""

    column_names = ['doc_id', 'title_kk', 'title_zh', 'url', 'passage', 'query_1_id', 'query_1', 'query_2_id', 'query_2']
    data_original = pd.DataFrame(columns=column_names)

    for index, row in passages_df.iterrows():

        dit = {}
        dit["doc_id"] = row["doc_id"]
        dit["title_kk"] = row["title_kk"]
        dit["title_zh"] = row["title_zh"]
        dit["url"] = row["url"]

        passages = list(row[4:].values)
        for p in passages:
            if not pd.isna(p):
                # print(Prompt + p)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        # {"role": "system", "content": Prompt},
                        {"role": "user", "content": p + Prompt}
                    ]
                )

                unicode_string = response.choices[0].message.content
                
                # 如果返回的内容为空  直接进行下一个
                # if unicode_string.isspace():
                    # continue

                lines = unicode_string.split('\n')
                queries = query_reorganize(lines)
                query_1 = queries[0]
                query_2 = queries[1]

                dit["passage"] = p
                dit["query_1_id"] = random_id(query_1)
                dit["query_1"] = query_1
                dit["query_2_id"] = random_id(query_2)
                dit["query_2"] = query_2

                data_original = data_original._append(dit, ignore_index=True)
                
                # 延迟
                time.sleep(30)
        
        print(f'The {index}th data has been processed')

    data_original.to_excel("./data_original_1.xlsx", index=False)
    print("query generation completed")

    data_translate = query_translate(data_original)
    data_translate.to_excel("./data_translate.xlsx", index=False)
    print("query translate completed")

