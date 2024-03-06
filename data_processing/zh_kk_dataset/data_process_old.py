import pymysql
from bs4 import BeautifulSoup
import pandas as pd



# 数据库连接信息
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "yh980727",
    "database": "total_data_4"
}
# 建立数据库链接
conn = pymysql.connect(**db_config)
# 创建一个游标对象
cursor = conn.cursor()
# SQL查询语句
sql_query = "SELECT paper_id, paper_title, paper_text, paper_url FROM total_data_copy1 LIMIT 50"
# 执行查询
cursor.execute(sql_query)
# 获取所有查询结果
# 获取的 results 是一个元组，每条数据又是一个元组
results = cursor.fetchall()

column_names = ['doc_id', 'title', 'text', 'url',
                'passage_1', 'passage_2', 'passage_3', 'passage_4', 'passage_5',
                'passage_6', 'passage_7', 'passage_8', 'passage_9', 'passage_10',
                'passage_11', 'passage_12', 'passage_13', 'passage_14', 'passage_15',
                'passage_16', 'passage_17', 'passage_18', 'passage_19', 'passage_20']
df = pd.DataFrame(columns=column_names)

# 处理查询结果
for row in results:

    dit = {}
    dit["doc_id"] = row[0]
    dit["title"] = row[1]

    # 拿出含有标签的文本
    original_text_with_tag = row[2]  # 通过字段名
    soup = BeautifulSoup(original_text_with_tag, 'html.parser')
    # text_with_tag = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6' 'p'])
    text_with_tag = soup.find_all(['p'])

    paragraphs = []
    for p in text_with_tag:
        paragraphs.append(p.text)
    text = ".".join(paragraphs)
    dit["text"] = text
    dit["url"] = row[3]

    n = 0
    for i, passage in enumerate(paragraphs):
        if n <= 20 :
            key = f'passage_{i}'
            dit[key] = passage
            n = n + 1
        else:
            break
    df = df._append(dit, ignore_index=True)

df.to_excel('text.xlsx', index=False)

# 关闭游标和数据库连接
cursor.close()
conn.close()
