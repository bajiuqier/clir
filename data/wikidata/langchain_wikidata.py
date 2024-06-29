from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

# 初始化 API 包装器
api_wrapper = WikidataAPIWrapper()
api_wrapper.doc_content_chars_max = 4000
api_wrapper.lang = "zh"
api_wrapper.top_k_results = 8
# 初始化查询运行器
wikidata_query_run = WikidataQueryRun(api_wrapper=api_wrapper)

# 执行查询
result = wikidata_query_run.run("中国")
print(result)
