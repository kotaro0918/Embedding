# Embedding in python

langchain の embedding をpythonで試行した結果報告




- [Embedding in python](#embedding-in-python)
  - [langchain をcontainer内で使う準備](#langchain-をcontainer内で使う準備)
  - [doc\_class.pdf を読み込ませる](#doc_classpdf-を読み込ませる)
    - [pdfをページごとに分割した場合](#pdfをページごとに分割した場合)
    - [要素ごとに分割した場合](#要素ごとに分割した場合)
    - [pdfをHTMLとして読み込む](#pdfをhtmlとして読み込む)
  - [doc\_class.txt を読み込ませる](#doc_classtxt-を読み込ませる)
    - [出力を１つに指定した場合と複数に指定した場合](#出力を１つに指定した場合と複数に指定した場合)
  - [全体を通したまとめ](#全体を通したまとめ)


## langchain をcontainer内で使う準備

1. 作業用ディレクトリにこのレポジトリをクローンする

1. clone したローカル環境下に　.devcontainer/devcontainer.envのファイルを作り
```- [Embedding in python](#embedding-in-python)
  - [langchain をcontainer内で使う準備](#langchain-をcontainer内で使う準備)
  - [doc\_class.pdf を読み込ませる](#doc_classpdf-を読み込ませる)
    - [pdfをページごとに分割した場合](#pdfをページごとに分割した場合)
    - [要素ごとに分割した場合](#要素ごとに分割した場合)
    - [pdfをHTMLとして読み込む](#pdfをhtmlとして読み込む)
  - [doc\_class.txt を読み込ませる](#doc_classtxt-を読み込ませる)
    - [出力を１つに指定した場合と複数に指定した場合](#出力を１つに指定した場合と複数に指定した場合)
  - [全体を通したまとめ](#全体を通したまとめ)

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
全体のコードはpdf_pages.pyにあります
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

### 要素ごとに分割した場合
.コード  
全体のコードはpdf_unstructured.pyにあります
```
from langchain.document_loaders import UnstructuredFileLoader
target_input=input()
loader = UnstructuredFileLoader("doc_class.pdf",mode="elements")
docs = loader.load()
print(f"number of docs: {len(docs)}")
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
```
上記のコードによってpdfを要素（ある程度の意味を持った一区切り）ごとに分割し、embeddingをかけています  
UnstructuredFileLoaderのmodeをsingleにするとpdfを丸ごと読み込みます

今回は3408個にpdfを分割して、読み込んでいます

・結果  
ターゲット文章
```
近代以前できた根室本線は北海道の滝川駅から帯広、釧路を経て根室駅を結ぶＪＲ北海道の路線です。このうち釧路駅から根室駅までの区間は「花咲線」の愛称で呼ばれています。観光シーズンには札幌からのリゾート列車が多数運行されます。キハ283系の車体は、ブルーとグリーンに丹頂鶴の赤を組み合わせ北海道らしさを演出しています.
```

```
以下の文章は本の解説です。この情報をもとにこの本に適した分類項目を三桁の数字で示してください
text: 鳥取県の白兎海岸は、日本最古の歴史書「古事記」に記された神話「因幡の白兎」の舞台と言われています。その神話に登場するワニを食べる伝統が２００キロほど離れた中国山地の広島県三次市に残されています。この「ワニ」とは、実はサメのこと。三次では昔から、お正月をはじめ、めでたいことがあるとワニの刺身を食べてきました。今は九州などから運ばれて来ます。地元で食品会社を経営する藤田恒造さんが仕入れるワニはおいしいと評判。妻と娘は、ワニの肉を使ったアイデア料理も開発しています。その１つ、娘の佳江さんがつくった照り焼き風味のワニバーガー。三次でもワニを食べたことのない子どもが増えていることに危機感を抱き、発案しました。
```

出力
```
{'output_text': 'この本に適した分類項目は、686です。'}
Tokens Used: 396
        Prompt Tokens: 379
        Completion Tokens: 17
Successful Requests: 1
Total Cost (USD): $0.0006025
```

```
{'output_text': 'この本に適した分類項目は596.63です。'}
Tokens Used: 611
        Prompt Tokens: 593
        Completion Tokens: 18
Successful Requests: 1
Total Cost (USD): $0.0009255
```

有用性及び問題点  

今回は要素ごとに分割したぶん、参照する情報に割くtoken数が非常に少なく抑えることができ、全体のtoken数もpageごとに分割したものと比べ、かなり低く抑えられる  
問題点としては、要素ごとに分割したときの分割点が適切でないことに起因する、参照箇所の不備が分類項目のずれにも影響してきてしまう点である  
例えば
```
219.9 沖縄県［琉球国］
```
219の下に含まれる219.9は考古学に関する遺跡のうち沖縄県にあるものをさすが、上の説明では沖縄県に関するもの全てをさすと解釈されてしまう

### pdfをHTMLとして読み込む
・コード  
全体のコードはpdf_miner.pyとpdfminer_experiment.pyにあります

```
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
loader = PDFMinerPDFasHTMLLoader("doc_class.pdf")
data = loader.load()[0]
```
上のコードによってpdfをHTML化して読み込んでいます  
・有用性及び問題点

現状では読み込んだhtmlから情報を抜き出して、分割することが難しく、このままembeddingをかけるとmaxtokenoverになる点が問題  
HTMLのコードの差を利用して分割することができれば、かなり使いやすいものとなりそう。
BeautifulSoupを使ってみたり、違うpdfであれば使えるかもしれない



##　doc_class.txtを読み込ませる

## doc_class.txt を読み込ませる
・コード  
全体のコードはtxt_split.py にあります
```
loader = TextLoader('doc_class.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
```

上記のコードによってtxtを500tokenごとに分割し、chunk_overlapを50に設定することで、前後の文脈の補完を行ってます  

CharcterTextSolitter では改行コードを区切り文字として、txtを分割した後、chunksizeの条件に合うように一区画を形成する


 
・結果  


ターゲット文章1
```
近代以降できた根室本線は北海道の滝川駅から帯広、釧路を経て根室駅を結ぶＪＲ北海道の路線です。このうち釧路駅から根室駅までの区間は「花咲線」の愛称で呼ばれています。観光シーズンには札幌からのリゾート列車が多数運行されます。キハ283系の車体は、ブルーとグリーンに丹頂鶴の赤を組み合わせ北海道らしさを演出しています.
```
ターゲット文章２
```
大阪市浪速区の繁華街・新世界に立つ、大阪のシンボル「通天閣」。現在の通天閣は、実は二代目。初代の通天閣は、明治４５年に建設され、町は賑わいましたが、昭和１８年、火災のため、失われてしまいます。その１０年後、新世界の人々は、通天閣再建のために立ち上がります。
```



出力１
```
{'output_text': '1. 682 交通史・事情\n2. 685.2 陸運史・事情\n3. 686.5 旅客\n\n根拠: 本文には根室本線の歴史や特徴が述べられており、交通史や陸上交通史の分類に適しています。また、観光シーズンやリゾート列車の情報も含まれており、旅客の分類にも適しています。'}
Tokens Used: 3384
        Prompt Tokens: 3240
        Completion Tokens: 144
Successful Requests: 1
Total Cost (USD): $0.005148000000000001
```
出力２

```
{'output_text': '210.12 文化史'}
Tokens Used: 2920
        Prompt Tokens: 2913
        Completion Tokens: 7
Successful Requests: 1
Total Cost (USD): $0.0043835
```

有用性及び問題点

今回の方法では、token数を抑えつつ参照の不備もかなり少ないので、分類に非常に使いやすい方法であった。  
上二つの文章の分類を比較してもかなり近い分類項目が出力されている。

chunk_sizeの試行として、1000tokenと300tokenを試したが、前者ではmaxtokenove後者では区切りの位置がおかしくなってしまった

またchunk_overlapを設定しないと前後の文脈補完ができなくなる


### 出力を１つに指定した場合と複数に指定した場合  


ターゲット文章１
```
対馬の北端にある比田勝(ひたかつ)は、プサンまでわずか５０キロ。日本でもっとも韓国に近い港です。御年９１歳の佐伯哲生さんは、タマネギ農家を営みながら、お隣プサンの街の風景を写真に収めています。韓国と佐伯さんを繋いだのはタマネギ。昭和１０年、９歳の頃に佐伯さんは、おじいさんと一緒にプサンにタマネギを売りに行きました。プサンは、日本と朝鮮半島を結ぶ拠点として開発が進み、映画館や大きな病院もあり、対馬の人々にとって憧れの場所でした。しかし日本の敗戦後、韓国との国交は途絶え、佐伯さんが再びプサンを訪れたのは戦後４０年近く過ぎたころ。以来、２年に１度は海を渡り、プサンの写真を撮り続けています。
```

複数個の場合
```
{'output_text': 'この本は、対馬の北端にある比田勝（ひたかつ）と韓国のプサンを結ぶ関係を描いています。また、御年９１歳の佐伯哲生さんがタマネギ農家を営みながら、プサンの風景を写真に収めていることも述べられています。この情報をもとに、この本に適した分類項目は以下の通りです。\n\n- 219.9 沖縄県［琉球国］（対馬の地域に関連する）\n- 221 朝鮮（韓国との関係に関連する）\n- 210.088 史料．日記．古文書（佐伯さんの写真に関連する）'}
Tokens Used: 2112
        Prompt Tokens: 1883
        Completion Tokens: 229
Successful Requests: 1
Total Cost (USD): $0.0032825000000000003
```

一つに限った場合
```
{'output_text': 'この本は、対馬の北端にある比田勝（ひたかつ）と韓国のプサンの関係を描いたものです。そのため、適した分類項目は213.61です。'}
Tokens Used: 1952
        Prompt Tokens: 1885
        Completion Tokens: 67
Successful Requests: 1
Total Cost (USD): $0.0029615
```

ターゲット文章２
```
東京都内から南へ２９０キロの海に浮かぶ八丈島。縄文時代に人々が住み始めたという島に、重要な言葉が残されています。八丈島の島ことば。日本に残る貴重な方言の１つで、「万葉集東歌」の文法の特徴と共通点があります。「万葉集東歌」は、奈良時代、関東周辺の人々が詠んだ歌で、形容詞の語尾が「悲しけ」、「長け」と、「け」で終わるなどの特徴があり、八丈の島ことばも同じ。ユネスコの調査により、八丈の島ことばは「消滅の危機にある言語」とされています。町では、島ことばを子どもたちに伝え親しんでもらう活動を続けています。その一つが、島ことばのカルタ。 言葉を覚えるため、絵札も字札も拾う特別ルールです。島民みんなの誇り、島ことばです。
```

複数個の場合
```
{'output_text': '910.4 日本文学\n210.12 文化史'}
Tokens Used: 3171
        Prompt Tokens: 3156
        Completion Tokens: 15
Successful Requests: 1
Total Cost (USD): $0.004764
```

一つに限った場合
```
{'output_text': '910.49'}
Tokens Used: 3026
        Prompt Tokens: 3023
        Completion Tokens: 3
Successful Requests: 1
Total Cost (USD): $0.0045405
```

両者の比較  
複数個の方が一つに限った場合と比べて、適切な分類項目を出力することが多い  
(これは一つに限るという制約が変に影響しているからなのでは)  

token数を単純に比較した場合では、一つに限った方が少ない

また、どちらの方法にもみられた傾向として、プロンプトの中に分類の根拠を合わせて  
出力すると表記することでより適切なものが出力される  


## 全体を通したまとめ

今回試行してみた方法の中では、ChracterTextSplitterが一番使いやすいように感じた。

理由としては分割の範囲を自由度高く指定できるのに加え、前後の文脈を損なわないようにchunk_overlapを設定できる方である

この方法である程度の文章の分類をすることができるのだが、全体を通した問題点として、chatgptの語句の解釈にぶれが出てしまうことがある

例えば
```
千葉県香取市の「十六島(じゅうろくしま)」は、利根川上流からの土砂が堆積してできた水田地帯。江戸時代から続く水郷の風景を舟に乗って楽しむことができます。案内をするのは女船頭たち。笹の葉に似せたサッパ舟を、棹（さお）一本で操ります。十六島は、江戸時代から農地として開拓され、エンマと呼ばれる水路が張り巡らされました。農家が米俵の運搬に使ったサッパ舟は、５０年ほど前まで一家に三艘(そう)はあったといいます。しかし昭和３９年に始まった土地改良事業によって、エンマは一部を残して埋め立てられました。女船頭のひとり、髙塚すぎさんは、故郷の伝統を残そうと後継者を育成しています
```
上記の文章をターゲットにした場合、出力は

```
{'output_text': '1. 210.17 災異史\n根拠: 本のタイトルに「守り継がれる」という言葉があり、十六島の風景が土地改良事業によって変わったことが述べられています。この事業は災害とも言える出来事であり、災異史の分類に適しています。\n\n2. 290 地理．地誌．紀行\n根拠: 本の内容は十六島の地理や風景について述べられています。また、舟に乗って楽しむことができるとも述べられており、地理や地誌の分類に適しています。\n\n3. 723.1 日本の洋画\n根拠: 本のタイトルに「原風景」という言葉があり、十六島の風景が描かれている可能性があります。このような風景画は日本の洋画の分類に適しています。'}
Tokens Used: 3367
        Prompt Tokens: 3063
        Completion Tokens: 304
Successful Requests: 1
Total Cost (USD): $0.0052025000000000005
```

出力の中の根拠の部分に注目すると、原風景→風景画のように解釈の飛躍が起きてしまっている部分がある.  
この部分に対して、タイトルの情報やみちしる内でのキーワードなどを与えてみたが、改善は見られなかった。

このような解釈の飛躍を防ぐことが、今後の課題になっていくと思う
