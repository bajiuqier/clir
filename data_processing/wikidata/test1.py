import requests

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

# 发送请求
response = requests.get(url, params=params)

# 检查请求是否成功
if response.status_code == 200:
    # 获取JSON响应
    data = response.json()
    print(data)
    
    # 处理响应数据
    for result in data['results']['bindings']:
        item = result['item']['value'].split('/')[-1]
        prop = result['p']['value'].split('/')[-1]
        adjItem = result['adjItem']['value'].split('/')[-1]
        
        print(f'Item: {item}')
        print(f'Property: {prop}')
        print(f'adjItem: {adjItem}')
        print()
else:
    print(f'Request failed with status code: {response.status_code}')