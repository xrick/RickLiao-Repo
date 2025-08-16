# 使用比喻法淺顯解讀 RAG 分塊策略，並以 Python 範例具體應用

以下內容以「整理書籍章節」「切分食譜步驟」「旅遊行程規劃」「郵件分發」「知識地圖」等日常生活場景為比喻，說明各種 RAG 分塊策略核心理念，並附上可實作的 Python 範例，展示如何在實務中運用。

---

## 1. 固定大小分塊（Fixed-Size Chunking）

比喻：像把一大捆信封按每十封一束分好，方便批次投遞。
Python 範例：將長篇電子郵件清單，每 100 行為一個分塊， 준비 分批寄送。

```python
def fixed_size_chunks(file_path, chunk_size=100):
    """
    將長郵件清單按每 chunk_size 行拆分，
    並分別輸出成新的檔案以便分批寄送。
    """
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i+chunk_size]
        out_name = f"emails_part_{i//chunk_size+1}.txt"
        with open(out_name, 'w', encoding='utf-8') as fout:
            fout.writelines(chunk)
        print(f"已產生 {out_name}，共 {len(chunk)} 封郵件")
```

---

## 2. 遞歸字元文本分割（Recursive Character Text Splitting）

比喻：像先依照章節標題拆書，再遇到章節過長時，再以小節、段落依序分割，保留自然脈絡。
Python 範例：對一篇旅遊攻略文章，先以「## 」作為大節，再以「### 」和空行分割，最後確保每段不超過 500 字。

```python
import re

def recursive_split(text, max_len=500):
    # 大節拆分
    parts = re.split(r'(## .+)', text)
    chunks = []
    for idx in range(1, len(parts), 2):
        header, content = parts[idx], parts[idx+1]
        if len(content) <= max_len:
            chunks.append(header + content)
        else:
            # 小節拆分
            subparts = re.split(r'(### .+)', content)
            for j in range(1, len(subparts), 2):
                sub_header, sub_text = subparts[j], subparts[j+1]
                # 以空行拆段落
                paras = sub_text.split('\n\n')
                cur = sub_header
                for p in paras:
                    if len(cur) + len(p) <= max_len:
                        cur += p + '\n\n'
                    else:
                        chunks.append(cur)
                        cur = sub_header + p + '\n\n'
                if cur:
                    chunks.append(cur)
    return chunks

# 使用範例
with open('travel_guide.md', encoding='utf-8') as f:
    txt = f.read()
for i, c in enumerate(recursive_split(txt)):
    print(f"Chunk {i+1}: {len(c)} 字元")
```

---

## 3. 語義分塊（Semantic Chunking）

比喻：像將食譜按照「配料」「醃製」「烹調」「擺盤」等步驟分段，而非按文字長度。
Python 範例：對 Markdown 格式的食譜，先斷句後計算句向量相似度，當相似度低於閾值即切分。

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_chunks(sentences, threshold=0.7):
    embeddings = model.encode(sentences)
    chunks, cur = [], [sentences[0]]
    prev_emb = embeddings
    for sent, emb in zip(sentences[1:], embeddings[1:]):
        sim = np.dot(prev_emb, emb) / (np.linalg.norm(prev_emb)*np.linalg.norm(emb))
        if sim < threshold:
            chunks.append(' '.join(cur))
            cur = [sent]
        else:
            cur.append(sent)
        prev_emb = emb
    chunks.append(' '.join(cur))
    return chunks

# 使用範例
with open('recipe.md', encoding='utf-8') as f:
    sents = [s.strip() for s in f.read().split('。') if s]
for i, chunk in enumerate(semantic_chunks(sents)):
    print(f"步驟區段 {i+1}: {chunk[:50]}…")
```

---

## 4. 標記優化分塊（Token-Based Chunking）

比喻：像以準確的包裹重量限制來分箱，不多也不少，充分利用每個箱子空間。
Python 範例：對文字檔案使用 tiktoken 計算 OpenAI 標記，保證每塊都不超過 512 token。

```python
import tiktoken

def token_chunks(text, max_tokens=512):
    enc = tiktoken.get_encoding('cl100k_base')
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end
    return chunks

# 使用範例
with open('chat_history.txt', encoding='utf-8') as f:
    content = f.read()
for idx, c in enumerate(token_chunks(content)):
    print(f"第 {idx+1} 塊：約 {len(c)} 字元")
```

---

## 5. 層次化分塊（Hierarchical Chunking）

比喻：像公司組織架構圖，從部門到小組再到個人，每層都清晰呈現。
Python 範例：以 JSON 文件為例，從最外層「章節」→「子章」→「段落」多層讀取並分塊。

```python
import json

def hierarchical_chunks(json_path):
    with open(json_path, encoding='utf-8') as f:
        doc = json.load(f)
    chunks = []
    for chapter in doc['chapters']:
        for section in chapter.get('sections', []):
            text = chapter['title'] + ' - ' + section['title'] + '\n' + section['content']
            chunks.append(text)
    return chunks

# 使用範例
for i, ch in enumerate(hierarchical_chunks('ebook_structure.json')):
    print(f"層次分塊 {i+1}: {ch.splitlines()[0]}")
```

---

## 6. 後期分塊（Late Chunking）

比喻：像先把整本書掃描成電子檔後，再按主題標籤分章，而非先拆頁再掃描。
Python 範例：使用長上下文嵌入模型先取整篇文章嵌入，然後在向量空間中以 KMeans 分群對應分塊。

```python
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch

# 載入長上下文模型
tokenizer = AutoTokenizer.from_pretrained('jina-ai/jina-embeddings-v2')
model = AutoModel.from_pretrained('jina-ai/jina-embeddings-v2')

def late_chunking(text, num_chunks=5):
    inputs = tokenizer(text, return_tensors='pt', truncation=False)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    kmeans = KMeans(n_clusters=num_chunks, random_state=42).fit(emb)
    return kmeans.labels_

# 使用範例
text = open('long_manual.txt', encoding='utf-8').read()
labels = late_chunking(text, num_chunks=4)
print("每個句子對應的分組標籤：", labels)
```

---

## 7. 上下文檢索分塊（Contextual Retrieval）

比喻：像針對每段旅遊介紹，先撰寫一句概括標語，再帶入詳細內容，方便讀者快速理解重點。
Python 範例：對新聞段落，用 GPT 生成摘要當作上下文，與原文合併後向量化。

```python
import openai

openai.api_key = 'YOUR_API_KEY'

def enrich_with_context(paragraphs):
    enriched = []
    for p in paragraphs:
        resp = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{"role":"user","content":f"為以下段落撰寫一句摘要：\n\n{p}"}]
        )
        summary = resp.choices[0].message.content.strip()
        enriched.append(summary + "\n\n" + p)
    return enriched

# 使用範例
news = open('news_paragraphs.txt', encoding='utf-8').read().split('\n\n')
for item in enrich_with_context(news):
    print(item[:80].replace('\n',' '))
```

---

## 8. 智能體分塊（Agentic Chunking）

比喻：像請一位編輯根據文章脈絡自行判斷最佳分段，再逐段校訂。
Python 範例：用 GPT 自動判斷切分點，將長篇說明書拆成若干獨立主題小節。

```python
import openai

def agentic_chunking(text):
    prompt = (
        "請將以下技術文件依據主題邏輯，自動拆分成多個有意義的小節，"
        "並在每個小節前加上標題：\n\n" + text
    )
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# 使用範例
manual = open('tech_manual.txt', encoding='utf-8').read()
print(agentic_chunking(manual)[:300])
```

---

透過上述八種策略及實務程式範例，您能根據文件類型、查詢需求與資源條件，靈活選擇與調整最適切的分塊方法，並在真實開發中快速上手。祝您在 RAG 系統建置中事半功倍！

來源
