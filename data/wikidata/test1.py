import requests
import pandas as pd

# 查询语句
query = """
SELECT ?item ?p ?adjItem
WHERE {
  VALUES (?item) {(wd:Q8739)}
  {
    ?item ?p ?adjItem .
    FILTER(isIRI(?adjItem))
    FILTER(STRSTARTS(STR(?p), 'http://www.wikidata.org/prop/direct/'))
  } UNION {
    ?adjItem ?p ?item .
    FILTER(isIRI(?adjItem))
    FILTER(STRSTARTS(STR(?p), 'http://www.wikidata.org/prop/direct/'))
  }
}
"""


url = 'https://query.wikidata.org/sparql'
params = {'query': query, 'format': 'json'}
data = []
# 发送请求
response = requests.get(url, params=params)

# 检查请求是否成功
if response.status_code == 200:
    # 获取JSON响应
    search_results = response.json()
    # print(search_results)
    
    # 处理响应数据
    for result in search_results['results']['bindings']:
        item = result['item']['value'].split('/')[-1]
        prop = result['p']['value'].split('/')[-1]
        adjItem = result['adjItem']['value'].split('/')[-1]
        data.append([item, prop, adjItem])

else:
    print(f'Request failed with status code: {response.status_code}')

print(data)
df = pd.DataFrame(data, columns=['Item', 'Property', 'AdjItem'])
print(df)


# 使用正则表达式过滤符合条件的行 匹配以 "Q" 开头后跟数字的字符串
# 前提是 AdjItem 列 的值是 str 类型的
df_filtered = df[df['AdjItem'].str.match(r'^Q\d+$')]
print(df_filtered)