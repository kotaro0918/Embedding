from langchain.embeddings import OpenAIEmbeddings

# テキストの準備
text = 'doc_class.txt'

# OpenAI APIによる埋め込みの生成
embeddings = OpenAIEmbeddings()
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])

# 確認
print(len(query_result))
print(query_result)
print(len(doc_result[0]))
print(doc_result)