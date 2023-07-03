
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("doc_class.pdf")
pages = loader.load_and_split()