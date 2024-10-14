import json
from fuzzywuzzy import fuzz

# 從文件加載知識庫
def load_knowledge_base(filename='knowledge_base.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# 保存知識庫到文件
def save_knowledge_base(knowledge_base, filename='knowledge_base.json'):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(knowledge_base, file, ensure_ascii=False, indent=4)

# RAG 系統：修正 ASR 輸出
def rag_correction(asr_output, context, knowledge_base):
    best_match = None
    best_score = 0
    threshold = 80  # 設定一個閾值，低於此值的匹配將被視為新詞

    for phrase, info in knowledge_base.items():
        # 計算字符串相似度
        base_score = fuzz.ratio(asr_output, phrase)
        
        # 計算上下文相關性
        context_score = sum(10 for c in context if c in info['context'])
        
        # 綜合得分
        total_score = base_score + context_score

        if total_score > best_score:
            best_score = total_score
            best_match = phrase

    if best_score < threshold:
        # 可能是新詞，添加到知識庫
        knowledge_base[asr_output] = {
            "correct": asr_output,
            "variants": [],
            "context": context
        }
        save_knowledge_base(knowledge_base)
        return asr_output, True  # 返回原始輸出和一個標誌，表示這是新詞
    
    return knowledge_base[best_match]['correct'], False

# 主程序
def main():
    knowledge_base = load_knowledge_base()
    
    # 如果知識庫為空，初始化它
    if not knowledge_base:
        knowledge_base = {
            "楊大先生": {
                "correct": "楊大先生",
                "variants": ["揚大先生", "陽大先生", "養大先生"],
                "context": ["名字", "人物", "先生"]
            }
        }
        save_knowledge_base(knowledge_base)

    # 模擬 ASR 輸出，包括一個新詞
    test_cases = [
        ("揚大先生", ["這是一個人名", "他是一位先生"]),
        ("陽大先生", ["這是一個人名", "他是一位教授"]),
        ("楊大先生", ["這是一個人名", "他是一位醫生"]),
        ("養大先生", ["這是一個人名", "他是一位工程師"]),
        ("羊大先生", ["這是一個人名", "他是一位企業家"])
    ]

    for asr_output, context in test_cases:
        corrected_output, is_new = rag_correction(asr_output, context, knowledge_base)
        if is_new:
            print(f"發現新詞: {asr_output}")
            print(f"上下文: {context}")
        else:
            print(f"ASR 輸出: {asr_output}")
            print(f"上下文: {context}")
            print(f"修正後: {corrected_output}")
        print("---")

    # 打印更新後的知識庫
    print("\n更新後的知識庫:")
    print(json.dumps(knowledge_base, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()
