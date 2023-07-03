from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

loader = TextLoader("doc_class.txt")
loader.load()