"""
1. 获取实体在 wikidata 中的对应 qid（或者自己在 wikidata 官网获取）
2，根据实体的 qid 获取其对应的三元组数据 (item_qid, relation_qid, adj_item_qid)
3. 然后根据获得 qid 获取对应的信息（英文、中文、哈萨克语的标签和描述信息）
"""

import requests

# 搁置 暂时用不到了