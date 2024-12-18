from pathlib import Path
import pandas as pd
# from translation_utils import google_translate
from tqdm import tqdm
import glob
import re
import os

HOME_DIR = Path(__file__).parent.parent / 'base_data'


def single_filling_item_info(item_info_file: str, filled_file: str = None,
                             save_filled_file: bool = True) -> pd.DataFrame:
    '''
    使用 谷歌翻译 API 翻译
    '''
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
            description_kk_trans = google_translate(text=description_en, dest=lang_kk, src=lang_src)
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


def multi_filling_item_info(item_info_file: str, filled_file: str = None,
                            save_filled_file: bool = True) -> pd.DataFrame:
    '''
    使用 谷歌翻译 API 翻译
    '''
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


def artificial_filling_item_info(item_info_file: str, filled_file: str):
    """
    将 过滤后的 item info 文件 存储为 xlsx 文件
    根据 label_en 和 description_en 手动翻译 过滤后的 item info
    将翻译后的 zh 和 kk info 添加 作为新的列数据 添加到 文件中
    添加的新列名 ["MT_label_zh", "MT_description_zh", "MT_label_kk", "MT_description_kk"]
    将zh kk 缺失的信息 根据新添加的数据 进行填充
    Parameters
    ----------
    item_info_file :
    filled_file :

    Returns
    -------

    """
    # item_info_df = pd.read_excel(item_info_file, index_col=0)
    # item_info_df = pd.read_excel(item_info_file)
    item_info_df = pd.read_csv(item_info_file, encoding='utf-8')

    for index, row in tqdm(item_info_df.iterrows(), total=item_info_df.shape[0]):
        if pd.isna(row["label_zh"]) and not pd.isna(row["MT_label_zh"]):
            row["label_zh"] = row["MT_label_zh"]

        if pd.isna(row["description_zh"]) and not pd.isna(row["MT_description_zh"]):
            row["description_zh"] = row["MT_description_zh"]

        if pd.isna(row["label_kk"]) and not pd.isna(row["MT_label_kk"]):
            row["label_kk"] = row["MT_label_kk"]

        if pd.isna(row["description_kk"]) and not pd.isna(row["MT_description_kk"]):
            row["description_kk"] = row["MT_description_kk"]

    item_info_df = item_info_df[
        ["qid", "label_zh", "label_kk", "label_en", "description_zh", "description_kk", "description_en"]]
    item_info_df.to_csv(filled_file, index=False, encoding='utf-8')
    print("--------------------------------------")
    print(f"填充好的数据已经存储在了{filled_file}")
    print("--------------------------------------")


if __name__ == "__main__":
    # item_info_file = str(HOME_DIR / 'base_test2_query_entity_info_filtered_MT_info.xlsx')
    # item_info_filled_file = str(HOME_DIR / 'base_test2_query_entity_info_filled.csv')

    # item_info_filled_df = single_filling_item_info(item_info_file=item_info_file, filled_file=item_info_filled_file, save_filled_file=True)
    # print(item_info_filled_df)
    # artificial_filling_item_info(item_info_file=item_info_file, filled_file=item_info_filled_file)

    folder_path = HOME_DIR.parent / 'dididi' / 'base_train_adj_item_info'
    #
    # # 获取所有匹配的文件
    # all_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    #
    # #
    # pattern = r'base_train_adj_item_info_filtered_\d+\.xlsx'
    #
    # # 筛选符合模式的文件
    # matched_files = [f for f in all_files if re.match(pattern, os.path.basename(f))]
    #
    # # 按文件名中的数字排序
    # matched_files.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
    #
    # # 读取并合并所有CSV文件
    # df_list = [pd.read_excel(excel_file) for excel_file in matched_files]
    # combined_df = pd.concat(df_list, ignore_index=True)

    adj_item_info_MT = str(folder_path / 'base_train_adj_item_info_MT.csv')
    # 将合并后的数据保存为新的CSV文件
    # combined_df.to_csv(adj_item_info_MT, index=False, encoding='utf-8')

    # print(f"合并完成，输出文件: {adj_item_info_MT}")

    item_info_filled_file = str(folder_path / 'base_train_adj_item_info_filled.csv')
    # artificial_filling_item_info(item_info_file=adj_item_info_MT, filled_file=item_info_filled_file)
    print(item_info_filled_file)