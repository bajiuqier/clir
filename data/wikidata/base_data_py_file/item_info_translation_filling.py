from pathlib import Path
import pandas as pd
from translation_utils import google_translate
from tqdm import tqdm

HOME_DIR = Path(__file__).parent.parent / 'base_data'

def filling_item_info_old(item_info_file: str, filled_file: str=None, save_filled_file: bool=True) -> pd.DataFrame:
    item_info_df = pd.read_csv(item_info_file, encoding='utf-8')

    lang_zh = 'zh-cn'
    lang_kk = 'kk'
    lang_src = 'en'

    error_str = "error error error la"
    has_error_num = 0

    for index, row in tqdm(item_info_df.iterrows(), total=item_info_df.shape[0]):
        label_en = row['label_en']
        description_en = row['description_en']

        # 填充中文标签
        if pd.isna(row['label_zh']) or row['label_zh'] == error_str:
            label_zh_trans = google_translate(text=label_en, dest=lang_zh, src=lang_src)
            if label_zh_trans == error_str:
                has_error_num += 1
            else:
                item_info_df.at[index, 'label_zh'] = label_zh_trans
                has_error_num = 0

        # 填充哈萨克文标签
        if pd.isna(row['label_kk']) or row['label_kk'] == error_str:
            label_kk_trans = google_translate(text=label_en, dest=lang_kk, src=lang_src)
            if label_kk_trans == error_str:
                has_error_num += 1
            else:
                item_info_df.at[index, 'label_kk'] = label_kk_trans

        # 填充中文描述
        if pd.isna(row['description_zh']) or row['description_zh'] == error_str:
            description_zh_trans = google_translate(text=description_en, dest=lang_zh, src=lang_src)
            if description_zh_trans == error_str:
                has_error_num += 1
            else:
                item_info_df.at[index, 'description_zh'] = description_zh_trans

        # 填充哈萨克文描述
        if pd.isna(row['description_kk']) or row['description_kk'] == error_str:
            description_kk_trans = google_translate(text=description_en, dest=lang_zh, src=lang_src)
            if description_kk_trans == error_str:
                has_error_num += 1
            else:
                item_info_df.at[index, 'description_kk'] = description_kk_trans
            
        if has_error_num >= 4:
            print("--------------------------------------------------------------")
            print(f"在第{index}行 出现全部数据翻译错误 结束翻译 之前翻译的数据仍会保存")
            print("更改相应的文件 重新执行翻译填充即可")
            print("--------------------------------------------------------------")
            break

    if save_filled_file:
        item_info_df.to_csv(filled_file, index=False, encoding='utf-8')

    return item_info_df

def filling_item_info(item_info_file: str, filled_file: str=None, save_filled_file: bool=True) -> pd.DataFrame:
    item_info_df = pd.read_csv(item_info_file, encoding='utf-8')

    lang_zh = 'zh-cn'
    lang_kk = 'kk'
    lang_src = 'en'

    # 填充中文标签
    print('----------------填充中文标签----------------')
    labels_to_translate_zh = item_info_df[pd.isna(item_info_df['label_zh'])]['label_en'].to_list()
    translated_labels_zh = google_translate(labels_to_translate_zh, dest=lang_zh, src=lang_src)
    if translated_labels_zh is not None:
        item_info_df.loc[pd.isna(item_info_df['label_zh']), 'label_zh'] = translated_labels_zh

    # 填充哈萨克文标签
    print('----------------填充哈萨克文标签----------------')
    labels_to_translate_kk = item_info_df[pd.isna(item_info_df['label_kk'])]['label_en'].to_list()
    translated_labels_kk = google_translate(labels_to_translate_kk, dest=lang_kk, src=lang_src)
    if translated_labels_kk is not None:
        item_info_df.loc[pd.isna(item_info_df['label_kk']), 'label_kk'] = translated_labels_kk

    # 填充中文描述
    print('----------------填充中文描述----------------')
    descriptions_to_translate_zh = item_info_df[pd.isna(item_info_df['description_zh'])]['description_en'].to_list()
    translated_descriptions_zh = google_translate(descriptions_to_translate_zh, dest=lang_zh, src=lang_src)
    if translated_descriptions_zh is not None:
        item_info_df.loc[pd.isna(item_info_df['description_zh']), 'description_zh'] = translated_descriptions_zh

    # 填充哈萨克文描述
    print('----------------填充哈萨克文描述----------------')
    descriptions_to_translate_kk = item_info_df[pd.isna(item_info_df['description_kk'])]['description_en'].to_list()
    translated_descriptions_kk = google_translate(descriptions_to_translate_kk, dest=lang_kk, src=lang_src)
    if translated_descriptions_kk is not None:
        item_info_df.loc[pd.isna(item_info_df['description_kk']), 'description_kk'] = translated_descriptions_kk

    if save_filled_file:
        item_info_df.to_csv(filled_file, index=False, encoding='utf-8')

    return item_info_df

if __name__ == "__main__":

    item_info_file = str(HOME_DIR / 'base_train_query_entity_filtered_info.csv')
    item_info_filled_file = str(HOME_DIR / 'base_train_query_entity_filled_info_1.csv')

    item_info_filled_df = filling_item_info_old(item_info_file=item_info_file, filled_file=item_info_filled_file, save_filled_file=True)
    print(item_info_filled_df)
    