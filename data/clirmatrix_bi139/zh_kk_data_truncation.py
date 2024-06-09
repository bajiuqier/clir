import json
import random


def data_truncation_order(input_file: str, output_file: str, num_records_to_keep: int):
    '''
    input_file:
    output_file:
    num_records_to_keep: 截取的数据量大小
    顺序截取数据中 num_records_to_keep 大小的数据量
    '''
    # 打开原始文件和目标文件
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        # 逐行读取原始文件
        for i, line in enumerate(f_in):
            # 如果已经写入了10000条记录，则退出循环
            if i >= num_records_to_keep:
                break

            # 解析JSON行
            data = json.loads(line.strip())

            # 写入到目标文件中
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f'已提取并保存前 {num_records_to_keep} 条记录到文件 {output_file}。')


def data_truncation_random(input_file: str, output_file: str, num_records_to_keep: int):
    '''
    input_file:
    output_file:
    num_records_to_keep: 截取的数据量大小
    随机截取数据中 num_records_to_keep 大小的数据量
    '''
    # 打开原始文件和目标文件
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        # 读取原始文件的总行数
        total_lines = sum(1 for line in f_in)
        print(total_lines)

        # 将文件指针移到文件开头
        f_in.seek(0)

        # 生成随机行号的列表
        random_indexes = random.sample(range(total_lines), num_records_to_keep)

        # 逐行读取原始文件，并写入到目标文件中
        for i, line in enumerate(f_in):
            if i in random_indexes:
                # 解析JSON行
                data = json.loads(line.strip())
                # 写入到目标文件中
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f'已随机选择并保存10000条记录到文件 {output_file}。')


if __name__ == '__main__':
    input_file = '/root/JupyterWorkspace/datasets/zh-kk-train.jsonl'
    output_file_random = '/root/JupyterWorkspace/datasets/zh-kk-train-100-random.jsonl'
    output_file_order = '/root/JupyterWorkspace/datasets/zh-kk-train-100-order.jsonl'
    num_records_to_keep = 100
    # data_truncation_random(input_file, output_file_random, num_records_to_keep)
    # data_truncation_order(input_file, output_file_order, num_records_to_keep)
