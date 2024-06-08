# import requests

# query = """
# SELECT ?item ?itemLabel ?p ?v ?vLabel
# WHERE {
#   VALUES (?item) {(wd:Q656448)}
#   OPTIONAL {
#     ?item ?p ?v .
#     ?v rdfs:label ?vLabel .
#     FILTER(lang(?vLabel) IN ('en', 'zh', 'kk'))
#     FILTER(isLiteral(?v))
#   }
#   OPTIONAL {
#     ?item ?p ?v .
#     ?v rdfs:label ?vLabel .
#     FILTER(lang(?vLabel) IN ('en', 'zh', 'kk'))
#     FILTER(isIRI(?v))
#     BIND(STRAFTER(STR(?v), 'http://www.wikidata.org/entity/') AS ?vId)
#   }
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "zh". }
# }
# """

# url = 'https://query.wikidata.org/sparql'
# params = {'query': query, 'format': 'json'}

# # 发送请求
# response = requests.get(url, params=params)

# # 检查请求是否成功
# if response.status_code == 200:
#     # 获取JSON响应
#     data = response.json()

#     # 处理响应数据
#     for result in data['results']['bindings']:
#         item = result['item']['value'].split('/')[-1]
#         item_label = result['itemLabel']['value']
#         prop = result['p']['value'].split('/')[-1]
#         if 'vId' in result:
#             value = result['vId']['value']
#             value_label = result['vLabel']['value']
#             print(f'Item: {item} ({item_label})')
#             print(f'Property: {prop}')
#             print(f'Value: {value} ({value_label})')
#             print()
#         else:
#             value = result['v']['value']
#             value_label = result['vLabel']['value']
#             print(f'Item: {item} ({item_label})')
#             print(f'Property: {prop}')
#             print(f'Value: {value} ({value_label})')
#             print()
# else:
#     print(f'Request failed with status code: {response.status_code}')
