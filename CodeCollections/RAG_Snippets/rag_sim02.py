from fuzzywuzzy import fuzz
import json

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

# 修正 ASR 輸出
def correct_asr_output(asr_output, context, knowledge_base):
    best_match = None
    best_score = 0
    threshold = 80  # 設定一個閾值，低於此值的匹配將被視為新詞

    for correct_phrase, info in knowledge_base.items():
        score = fuzz.ratio(asr_output, correct_phrase)
        for variant in info['variants']:
            variant_score = fuzz.ratio(asr_output, variant)
            score = max(score, variant_score)

        if score > best_score:
            best_score = score
            best_match = correct_phrase

    if best_score < threshold:
        # 可能是新詞，添加到知識庫
        knowledge_base[asr_output] = {
            "variants": [],
            "context": context
        }
        save_knowledge_base(knowledge_base)
        return asr_output, True  # 返回原始輸出和一個標誌，表示這是新詞
    
    return knowledge_base[best_match]['correct'] if best_match else asr_output, False

# 主程序
def main():
    knowledge_base = load_knowledge_base()
    
    # 模擬 ASR 輸出，包括一個新詞
    asr_outputs = ["揚大先生", "陽大先生", "楊大先生", "養大先生", "羊大先生"]
    context = ["這是一個人名", "他是一位先生"]

    for asr_output in asr_outputs:
        corrected_output, is_new = correct_asr_output(asr_output, context, knowledge_base)
        if is_new:
            print(f"發現新詞: {asr_output}")
        else:
            print(f"ASR 輸出: {asr_output} -> 修正後: {corrected_output}")

    # 打印更新後的知識庫
    print("\n更新後的知識庫:")
    print(json.dumps(knowledge_base, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()
