# 檢索增強生成（RAG）分塊策略最新技術研究報告

## 摘要

檢索增強生成（RAG）系統的效能很大程度上取決於其分塊策略的品質。本研究報告全面分析了2024-2025年間RAG分塊策略的最新發展，涵蓋了從傳統的固定大小分塊到先進的語義分塊、後期分塊、上下文檢索等創新技術。研究發現，頁面級分塊在多數情況下表現最佳，而語義感知的分塊方法雖然計算成本較高，但能顯著提升檢索精確度和生成品質。

## 1. 引言

RAG系統透過結合外部知識檢索和語言模型生成，已成為現代AI應用的核心架構。分塊（chunking）作為RAG管道中的關鍵預處理步驟，直接影響檢索精確度、上下文連貫性和最終回應品質[1][2]。不當的分塊策略可能導致相關資訊丟失、上下文碎片化，以及計算資源浪費[3]。

隨著大型語言模型上下文視窗的擴展和多模態文件處理需求的增長，分塊策略也在不斷演進。本報告基於最新研究文獻和實務案例，系統性分析了當前最先進的分塊技術，並提供實務應用指導。

## 2. 傳統分塊策略

### 2.1 固定大小分塊

固定大小分塊是最直觀的方法，基於預定義的字元數、詞數或標記數將文本分割成統一大小的段落[1][4]。雖然實作簡單且計算效率高，但經常在句子中間截斷，破壞語義邊界。為緩解此問題，通常引入重疊機制，讓相鄰分塊共享部分內容[2]。

**優勢**：實作簡單、可預測的效能、批次處理友好
**劣勢**：破壞語義邊界、重要資訊可能分散於多個分塊
**適用場景**：同質化文件、快速原型開發、資源受限環境

### 2.2 遞歸字元文本分割

遞歸分塊採用層次化分隔符策略，首先嘗試在段落邊界分割，然後依序使用句子結尾、標點符號等更細粒度的分隔符[2][5]。此方法在保持目標分塊大小的同時，盡可能保留自然文本邊界。

LangChain的`RecursiveCharacterTextSplitter`是此方法的典型實現，當較大的段落超過目標大小時，會遞歸地使用更細粒度的分隔符進行進一步分割[2]。

## 3. 語義感知分塊策略

### 3.1 語義分塊

語義分塊透過分析文本的語義相似性來決定分割點，而非依賴固定長度或簡單的結構標記[6][7]。該方法首先將文本分割成句子，然後計算連續句子間的嵌入相似性，當相似性顯著下降時創建新的分塊邊界[8]。

Greg Kamradt提出的語義分塊演算法使用滑動視窗計算連續段落間的餘弦距離，透過統計方法（如第95百分位數）設定閾值來識別語義轉換點[8]。然而，最新研究質疑語義分塊是否值得其計算成本，實驗顯示其相對於固定大小分塊的效能提升並不一致[7]。

### 3.2 層次化分塊

層次化分塊根據文件的自然結構（如章節、段落、小節）組織內容[9]。此方法特別適用於結構化文件，如技術文件、學術論文和政策文件，因為其固有結構反映了邏輯內容關係[2]。

實現可使用文件解析工具提取結構元數據（如字體大小、邊界框），自動化此過程[2]。對於掃描文件，可能需要光學字元識別（OCR）來推斷層次結構。

### 3.3 上下文豐富分塊

上下文豐富分塊透過添加周圍內容的摘要或標題資訊來增強各個分塊[6]。Microsoft的視窗化摘要技術透過相鄰分塊的摘要豐富文本分塊，提供更廣泛的上下文，動態調整「視窗大小」以探索不同的上下文範圍[6]。

## 4. 最新進階分塊技術

### 4.1 後期分塊（Late Chunking）

後期分塊是2024年由JinaAI提出的創新方法，顛覆了傳統的「先分塊後嵌入」順序[10][11]。該技術首先對整個文件進行嵌入，然後在平均池化之前進行分塊，使每個分塊嵌入都能捕獲整個文本的資訊[10]。

**工作原理**：
1. 使用長上下文嵌入模型對整個文件進行標記級嵌入
2. 在嵌入空間中進行分塊操作
3. 對分塊後的標記嵌入進行池化得到最終的分塊向量

此方法利用jina-embeddings-v2等長上下文嵌入模型的能力，在所有檢索基準測試中都優於傳統分塊方法[10][11]。

### 4.2 上下文檢索（Contextual Retrieval）

Anthropic提出的上下文檢索技術透過LLM為每個分塊生成描述性上下文，然後將此上下文與原始分塊內容結合後進行嵌入[12][13]。這種方法特別適用於處理含有代詞或模糊引用的內容，能顯著提高檢索精確度。

**實施步驟**：
1. 對每個分塊使用LLM生成上下文描述
2. 將上下文描述與原始分塊內容合併
3. 對增強後的分塊進行嵌入和索引
4. 在檢索時使用完整的上下文資訊

雖然此方法在預處理階段增加了LLM調用成本，但能顯著減少檢索時的幻覺問題[12]。

### 4.3 智能體分塊（Agentic Chunking）

智能體分塊使用LLM分析內容以識別基於獨立性和語義連貫性的邏輯分割點[14]。該方法透過精心設計的提示詞指導LLM識別自然斷點，確保每個分塊都是完整、有意義的部分[14]。

雖然這種方法能產生高品質的語義分塊，但由於需要多次LLM調用，處理成本相當高[15][16]。

### 4.4 標記優化分塊

考慮到LLM以標記為基本處理單位，標記優化分塊策略專注於最大化標記使用效率[17][18]。此方法使用tiktoken等標記化工具精確控制分塊大小，確保充分利用模型的上下文視窗[17]。

相較於基於字元數的分塊，標記優化方法能將更多有效內容塞入相同的標記限制內，提高處理效率並降低成本[18]。

## 5. 多模態和專門化分塊策略

### 5.1 多模態文件分塊

現代文件通常包含文字、圖像、表格等多種內容類型，需要專門的分塊策略[19][20]。多模態分塊的三種主要方法包括：

**層次化分塊**：基於文件自然結構組織內容，如將圖像與其標題和解釋文本組合在一起[19]。

**模態特定分塊**：分別處理不同資料類型，使用spaCy處理文字、OpenCV處理圖像、Camelot處理表格，然後透過元數據維護跨模態引用[19]。

**上下文感知分組**：保持邏輯連接，例如將截圖、程式碼片段和警告說明組合成單一單元[19]。

Unstructured.io提供了多種多模態分塊策略，包括基礎策略（basic）、按標題策略（by_title）、按頁面策略（by_page）和按相似性策略（by_similarity）[20]。

### 5.2 基於知識圖譜的分塊

知識圖譜引導的RAG（KG²RAG）將分塊表示為圖中的節點，透過實體關係邊連接相關內容[21][22]。此方法支援多跳推理和複雜查詢處理，特別適用於知識密集型領域[23]。

**主要優勢**：
- 支援多跳推理和關係查詢
- 減少對分塊配置的敏感性
- 透過圖遍歷增強檢索精確度

Microsoft的GraphRAG框架和TOBUGraph等系統展示了此方法在實際應用中的效果[22]。

## 6. 效能評估與基準測試

### 6.1 評估指標

RAG分塊策略的評估涵蓋檢索和生成兩個層面的指標[24][25]：

**檢索指標**：
- 上下文召回率（Context Recall）：成功檢索的相關文件比例
- 上下文精確度（Context Precision）：檢索文件中相關的比例
- Hit@K：相關項目出現在前K個檢索結果中的比例

**生成指標**：
- 忠實度（Faithfulness）：生成回應與檢索上下文的一致性
- 答案相關性（Answer Relevance）：回應與查詢的相關程度
- 答案正確性（Answer Correctness）：與標準答案的匹配度

### 6.2 基準測試結果

NVIDIA的綜合評估顯示，頁面級分塊在多個資料集上實現了最高的平均準確率（0.648）和最低的標準差（0.107），表現出跨不同內容類型的一致性效能[26]。

**關鍵發現**：
- 頁面級分塊總體表現最佳
- 中等大小分塊（512-1024標記）通常優於極端大小
- 極小（128標記）和極大（2048標記）分塊效能較差
- 查詢特性影響最佳分塊大小選擇

不同資料集的最佳策略存在差異：金融文件中，FinanceBench在1024標記分塊下表現最佳，而KG-RAG在頁面級分塊下表現最佳[26]。

## 7. 實務應用指導
### 7.1 策略選擇框架

選擇適當的分塊策略需要考慮多個因素：

**內容類型**：
- 結構化文件（法律、學術）：文件級或層次化分塊
- 對話資料：句子級或小型固定分塊
- 技術手冊：多模態或頁面級分塊

**查詢模式**：
- 事實性查詢：較小分塊（256-512標記）
- 分析性查詢：較大分塊（1024標記）或頁面級分塊
- 複雜推理：基於知識圖譜的分塊

**資源約束**：
- 低計算資源：固定大小或遞歸分塊
- 高品質需求：語義分塊或智能體分塊
- 實時處理：預處理基礎分塊

### 7.2 最佳實務建議

1. **以頁面級分塊為基準**：作為預設選擇，然後根據具體需求調整[26]
2. **測試多種方法**：在實際資料上評估不同策略的效能[26]
3. **考慮計算成本**：平衡處理品質與資源消耗[7]
4. **保持可逆性**：特別是在GraphRAG應用中，確保分塊過程可逆[23]
5. **使用重疊策略**：在需要資訊連續性的場景中應用10-20%重疊[27]

## 8. 未來發展趋势

### 8.1 技術演進方向

**自適應智慧**：RAG系統將動態調整檢索深度和生成策略[28]。醫療助理可能會在關鍵診斷時優先考慮同行評議研究，而在一般諮詢時檢索更廣泛的患者友好摘要。

**多模態整合**：RAG將超越文字應用，整合文字、圖像和音頻[28]。教育領域的RAG導師可能結合視覺圖表與文字解釋，根據學生的學習風格客製化內容。

**聯邦學習**：分散式RAG系統將能在不洩露隱私的情況下從多樣化資料集學習[28]，這對金融等對資料敏感性要求極高的行業特別重要。

### 8.2 新興技術趨勢

**自我改進系統**：利用強化學習基於使用者互動精煉檢索策略[28]。法律助理可能學會在使用者持續偏好基於先例的論證時優先考慮判例法而非成文法。

**跨學科整合**：結合RAG與因果推理模型，系統不僅能檢索相關資料，還能分析因果關係[28]。在醫療保健中，這意味著不僅識別治療選項，還能基於患者病史分析可能的結果。

**永續性考量**：隨著RAG規模擴展，優化檢索演算法以減少計算開銷將變得至關重要[28]。稀疏索引和量化嵌入等技術可能使大規模部署更加永續。

## 9. 結論

RAG分塊策略的發展反映了從簡單文本分割向智慧語義處理的演進。雖然固定大小分塊仍在許多應用中有效，但語義感知方法、後期分塊和上下文檢索等先進技術正在重新定義檢索品質的標準。

**關鍵要點**：

1. **頁面級分塊**在大多數情況下提供最佳的整體效能，是推薦的起點
2. **語義分塊**雖然計算成本較高，但在需要高品質檢索的場景中價值明顯
3. **後期分塊**和**上下文檢索**代表了分塊技術的最新突破，特別適用於複雜文件
4. **多模態分塊**成為處理現代複雜文件的必要技術
5. **評估與測試**對於選擇最適合特定用例的策略至關重要

隨著大型語言模型能力的持續提升和企業對AI應用需求的增長，分塊策略將繼續演進，朝向更智慧、更高效、更適應性強的方向發展。成功的RAG實施需要在效能、成本和複雜性之間找到適當平衡，而本報告提供的框架和建議將有助於實務工作者做出明智的技術選擇。

來源
[1] 5 Chunking Strategies For RAG - Daily Dose of Data Science https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag
[2] 5 Chunking Strategies For RAG Applications - Airbyte https://airbyte.com/data-engineering-resources/chunk-text-for-rag
[3] Breaking up is hard to do: Chunking in RAG applications https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/
[4] 5 Chunking Strategies for Retrieval-Augmented Generation.md https://github.com/xbeat/Machine-Learning/blob/main/5%20Chunking%20Strategies%20for%20Retrieval-Augmented%20Generation.md
[5] 7 Chunking Strategies in RAG You Need To Know - F22 Labs https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/
[6] A Guide to Chunking Strategies for Retrieval Augmented Generation ... https://zilliz.com/learn/guide-to-chunking-strategies-for-rag
[7] [2410.13070] Is Semantic Chunking Worth the Computational Cost? https://arxiv.org/abs/2410.13070
[8] Evaluating Chunking Strategies for Retrieval - Chroma Research https://research.trychroma.com/evaluating-chunking
[9] Enhancing Retrieval Augmented Generation with Hierarchical Text ... https://arxiv.org/abs/2507.09935
[10] Late Chunking: Balancing Precision and Cost in Long Context ... https://weaviate.io/blog/late-chunking
[11] [PDF] Late Chunking: Contextual Chunk Embeddings Using Long-Context ... https://arxiv.org/pdf/2409.04701.pdf
[12] Two Killer n8n RAG Strategies (Late Chunking & Contextual Retrieval) https://www.theaiautomators.com/two-killer-n8n-rag-strategies/
[13] Two NEW n8n RAG Strategies (Anthropic's Contextual Retrieval ... https://www.youtube.com/watch?v=61dvzowuIlA
[14] A Deep-Dive into Chunking Strategy, Chunking Methods, and ... https://www.superteams.ai/blog/a-deep-dive-into-chunking-strategy-chunking-methods-and-precision-in-rag-applications
[15] A Guide to Chunking Strategies for Retrieval Augmented Generation ... https://www.sagacify.com/news/a-guide-to-chunking-strategies-for-retrieval-augmented-generation-rag
[16] Chunking Strategies for LLM Applications: The Complete Guide https://globalnodes.tech/blog/chunking-strategy-for-llm-application/
[17] Token-Based Chunking for LLMs: Maximize Your Input in 10 Lines ... https://www.youtube.com/watch?v=AseBShIN920
[18] Token optimization: The backbone of effective prompt engineering https://developer.ibm.com/articles/awb-token-optimization-backbone-of-effective-prompt-engineering/
[19] What are effective chunking strategies for multimodal documents? https://milvus.io/ai-quick-reference/what-are-effective-chunking-strategies-for-multimodal-documents
[20] Chunking PDFs and Multimodal Documents: Efficient Methods for ... https://blog.gopenai.com/chunking-pdfs-and-multimodal-documents-efficient-methods-for-handling-text-tables-and-images-for-467472f02d34
[21] Knowledge Graph-Guided Retrieval Augmented Generation - arXiv https://arxiv.org/abs/2502.06864
[22] TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM ... https://arxiv.org/html/2412.05447v2
[23] Improving LLM Accuracy: Graph-Based Retrieval and Chunking ... https://www.cognee.ai/blog/deep-dives/enhancing-llm-responses-with-graph-based-retrieval-and-advanced-chunking-techniques
[24] RAG Evaluation Metrics: Best Practices for Evaluating RAG Systems https://www.patronus.ai/llm-testing/rag-evaluation-metrics
[25] Evaluation of RAG Metrics for Question Answering in the Telecom ... https://arxiv.org/html/2407.12873v1
[26] Finding the Best Chunking Strategy for Accurate AI Responses https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/
[27] Chunk Twice, Retrieve Once: RAG Chunking Strategies Optimized ... https://infohub.delltechnologies.com/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/
[28] Retrieval-Augmented Generation (RAG): The Definitive Guide [2025] https://www.chitika.com/retrieval-augmented-generation-rag-the-definitive-guide-2025/
[29] MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented ... https://arxiv.org/html/2503.09600v1
[30] Chunking strategies for RAG tutorial using Granite - IBM https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai
[31] Chunking Optimization for Retrieval-Augmented Generation (RAG) https://www.squareboat.com/blog/chunking-optimization-for-retrieval-augmented-generation
[32] Which RAG Chunking and Formatting Strategy Is Best for Your App ... https://www.tigerdata.com/blog/which-rag-chunking-and-formatting-strategy-is-best
[33] Evaluating Advanced Chunking Strategies for Retrieval-Augmented ... https://arxiv.org/abs/2504.19754
[34] Mastering Chunking Techniques for LLM Applications in 2025 https://www.puppyagent.com/blog/Mastering-Chunking-Techniques-for-LLM-Applications-in-2025
[35] Chunking in RAG: Strategies for Optimal Text Splitting - Chitika https://www.chitika.com/understanding-chunking-in-retrieval-augmented-generation-rag-strategies-techniques-and-applications/
[36] Advanced Chunking and Search Methods for Improved Retrieval ... https://openaccess.cms-conferences.org/publications/book/978-1-964867-35-9/article/978-1-964867-35-9_194
[37] Chunking Strategies for LLM Applications - Pinecone https://www.pinecone.io/learn/chunking-strategies/
[38] 【Document Intelligence】使用Layout Model 實作Semantic Chunking https://www.charliewei.net/2024/08/implement-semantic-chunking-using-Document-ntelligence-layout-model.html
[39] Chunking Analysis: Which is the right chunking approach for your ... https://blog.lancedb.com/chunking-analysis-which-is-the-right-chunking-approach-for-your-language/
[40] Chunking Strategy for LLM Application: Everything You Need to Know https://aiveda.io/blog/chunking-strategy-for-llm-application
[41] Late Chunking vs Contextual Retrieval - KX https://kx.com/resources/ebook/late-chunking-vs-contextual-retrieval/
[42] LLM chunking | Redis https://redis.io/blog/llm-chunking/
[43] How can I go about creating knowledge graphs using chunks from ... https://www.reddit.com/r/LocalLLaMA/comments/1clyyy0/how_can_i_go_about_creating_knowledge_graphs/
[44] Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5
[45] Enhancing RAG with Multimodal Document Understanding - arXiv https://arxiv.org/html/2506.16035
[46] How to Use Knowledge Graphs for Retrieval-Augmented Generation ... https://hackernoon.com/how-to-use-knowledge-graphs-for-retrieval-augmented-generationwithout-a-graph-db
[47] Graph RAG: Navigating graphs for Retrieval-Augmented Generation ... https://www.elastic.co/search-labs/blog/rag-graph-traversal
[48] RAG: An Architectural Review and Strategic Outlook for 2025 https://www.linkedin.com/pulse/rag-architectural-review-strategic-outlook-2025-bal%C3%A1zs-feh%C3%A9r-bwzpf
[49] VectorChat/new-chunking-benchmarks - GitHub https://github.com/VectorChat/new-chunking-benchmarks
[50] Evaluating RAG applications with RAGAS - Pathway https://pathway.com/blog/evaluating-rag
[51] Benchmarking Evaluation of LLM Retrieval Augmented Generation https://arize.com/blog-course/evaluation-of-llm-rag-chunking-strategy/
[52] Building and Evaluating Advanced RAG Applications。建立與評估進 ... https://hackmd.io/@YungHuiHsu/H16Y5cdi6
[53] Evaluation of RAG pipelines with Ragas - Langfuse https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas
[54] rag_chunking_strategies_comprehensive.csv https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/8b883b617e3226394c741ce0e3148cdf/791bcc98-b7de-4014-9a51-22eee92448e7/fb763c58.csv
