import random
from fuzzywuzzy import fuzz

# 模擬知識庫
knowledge_base = {
    "楊大先生": {
        "correct": "楊大先生",
        "variants": ["揚大先生", "陽大先生", "養大先生"],
        "context": ["名字", "人物", "先生"]
    },
    # 可以添加更多條目
}

# 模擬 ASR 輸出
def simulate_asr_output(correct_phrase):
    variants = knowledge_base[correct_phrase]["variants"]
    return random.choice([correct_phrase] + variants)

# RAG 系統
def rag_correction(asr_output, context):
    best_match = None
    best_score = 0
    
    for phrase, info in knowledge_base.items():
        # 檢查模糊匹配分數
        score = fuzz.ratio(asr_output, phrase)
        
        # 檢查上下文相關性
        context_relevance = sum(1 for c in context if c in info["context"])
        
        # 結合模糊匹配分數和上下文相關性
        total_score = score + (context_relevance * 10)  # 給予上下文更多權重
        
        if total_score > best_score:
            best_score = total_score
            best_match = phrase
    
    return knowledge_base[best_match]["correct"] if best_match else asr_output

# 主程序
def main():
    correct_phrase = "楊大先生"
    context = ["這是一個人名", "他是一位先生"]
    
    for _ in range(5):  # 模擬多次識別
        asr_output = simulate_asr_output(correct_phrase)
        corrected_output = rag_correction(asr_output, context)
        
        print(f"ASR 輸出: {asr_output}")
        print(f"RAG 修正: {corrected_output}")
        print("---")

if __name__ == "__main__":
    main()
