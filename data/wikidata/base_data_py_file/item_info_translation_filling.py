import pandas as pd
from googletrans import Translator
import time

# pip install googletrans==4.0.0-rc1
# 这个库要求 httpx==0.13.3
# 但环境中的其他库可能需要更高版本的 httpx 所以为翻译任务单独创建一个环境

# print(googletrans.LANGUAGES)
# data_path = '/home/yanghe/Downloads/1001-1500.xlsx'
# data_df = pd.read_excel(data_path, header=None)

# en_data = data_df.iloc[:100,1]
# en_data.to_csv('/home/yanghe/Downloads/501-600_en.csv', index=False, header=False)
# data_df[1].to_excel('/home/yanghe/Downloads/1001-1500.xlsx', index=False, header=False)

# for index, row in data_df.iterrows():
#     print(str(row[1]))
#     if index == 3:

#         break

# 设置Google翻译服务地址
# translator = Translator(service_urls=[
#       'translate.google.com'
# ])
translator = Translator()
translation = translator.translate('A woman with a black shirt and tan apron is standing behind a counter in a restaurant .', dest='zh-cn').text

print(translation)
# zh_text = []

# for index, row in data_df.iterrows():

#     if index < 0:
#         continue

#     time.sleep(10)
#     translation = translator.translate(row[1], dest='zh-CN').text
    
#     if index == 10:
#         break

#     zh_text.append(translation)
    
# print(len(zh_text))