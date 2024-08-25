import pandas as pd
# pip install googletrans==4.0.0-rc1
# 这个库要求 httpx==0.13.3
# 但环境中的其他库可能需要更高版本的 httpx 所以为翻译任务单独创建一个环境
from googletrans import Translator
from translation_utils import google_translate
import time

from pathlib import Path
HOME_DIR = Path(__file__).parent.parent / 'base_data'


# for index, row in data_df.iterrows():

#     if index < 0:
#         continue

#     time.sleep(10)
#     translation = translator.translate(row[1], dest='zh-CN').text
    
#     if index == 10:
#         break

#     zh_text.append(translation)
    
# print(len(zh_text))