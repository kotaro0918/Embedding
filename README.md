# Embedding in python

langchain の embedding をpythonで試行した結果報告

## langchain をcontainer内で使う準備

1. 作業用ディレクトリにこのレポジトリをクローンする

1. clone したローカル環境下に　.devcontainer/devcontainer.envのファイルを作り
```
OPENAI_API_KEY=自分のAPI Key
```
を書き込む     

3. .devcontainer/devcontainer.jsonに
```
"runArgs": ["--env-file", ".devcontainer/devcontainer.env"]
```
を書き込む   


今回は　「日本十進分類法（NDC）分類基準」を読み込ませ、みちしる(https://www.nhk.or.jp/archives/chiiki/ )内にある文章を分類するタスクに取り組んだ

日本十進分類法（NDC）分類基準はdoc_class.pdfとdoc_class.txtの２種類を用意した


## doc_class.pdf を読み込ませる

### pdfをページごとに分割した場合
・コード   
このコードはpdf_pages.pyにあります
```
from langchain.document_loaders import PyPDFLoader
target_input=input()
loader = PyPDFLoader("doc_class.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(pages, embeddings)
```

上記のコードによってpdfをページごとに分割して、embeddingをかけています

・結果  
ターゲット文章
```
近代以前できた根室本線は北海道の滝川駅から帯広、釧路を経て根室駅を結ぶＪＲ北海道の路線です。このうち釧路駅から根室駅までの区間は「花咲線」の愛称で呼ばれています。観光シーズンには札幌からのリゾート列車が多数運行されます。キハ283系の車体は、ブルーとグリーンに丹頂鶴の赤を組み合わせ北海道らしさを演出しています.
```  

出力  
```
{'output_text': '526.68 運輸・交通・観光事業の建築'}
Tokens Used: 3949
        Prompt Tokens: 3925
        Completion Tokens: 24
Successful Requests: 1
Total Cost (USD): $0.0059355
```
有用性及び問題点   

 この方法ではpdfをページごとに分割するため、プロンプト内で参照する情報もページ単になる。  
 この影響でtoken数が他の方法に比べ大きくなりやすいので、複数のキーワードを含む文章ではmaxtokenoverとなる  
 今回のターゲット文章のようにキーワードが少なく複数分野に跨らない場合はかなり正確に分類項目を出してくれる。  
 より短い文章を分類する場合は使えるかもしれない

 ### 


