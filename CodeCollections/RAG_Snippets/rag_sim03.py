"""
擴展上下文處理： 目前，我們的程序僅僅檢查上下文中的關鍵詞是否存在。我們可以改進這一點，使用更高級的自然語言處理技術。
"""

import spacy

nlp = spacy.load("zh_core_web_sm")

def calculate_context_relevance(context, stored_context):
    context_doc = nlp(" ".join(context))
    stored_doc = nlp(" ".join(stored_context))
    return context_doc.similarity(stored_doc)

# 在 rag_correction 函數中：
context_score = calculate_context_relevance(context, info['context']) * 50
