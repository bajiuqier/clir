from googletrans import Translator
import pandas as pd

# 设置Google翻译服务地址
translator = Translator(service_urls=['translate.google.com'])
def translate(sentences: str) -> str:
    # result = translator.translate(sentences, src='kk', dest='zh-cn')  #dest:目标语言
    result = translator.translate(sentences, dest='zh-cn').text  #dest:目标语言
    return result
  
def query_translate(df: pd.DataFrame) -> pd.DataFrame:

    df[["query_1_zh", "query_2_zh"]] = None

    for index, row in df.iterrows():
        if not pd.isna(row.query_1_kk):
            df.loc[index, "query_1_zh"] = translate(row.query_1_kk)
        if not pd.isna(row.query_2_kk):
            df.loc[index, "query_2_zh"] = translate(row.query_2_kk)
    
    return df