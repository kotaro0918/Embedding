from langchain.document_loaders import TextLoader

loader = TextLoader("doc_class.txt")
loader.load()