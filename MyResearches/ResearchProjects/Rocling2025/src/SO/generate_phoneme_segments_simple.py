
"""
創建一個不依賴音頻載入的簡化版本
"""

import os
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk
import argparse

def generate_simple_phoneme_segments(dataset_path: str, output_path: str):
    """為資料集生成簡單的 phoneme_segments 欄位（不載入音頻）"""
    
    # 載入資料集
    print(f"載入資料集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    def add_simple_phoneme_segments(example):
        """為單個樣本生成簡單的 phoneme_segments"""
        try:
            # 獲取音素序列
            phoneme_sequence = example["cmu_ipa_phonetic_transcription"]
            phoneme_count = len(phoneme_sequence)
            
            # 生成簡單的等長分段
            # 假設每個音素平均佔 10 幀（可調整）
            frames_per_phoneme = 10
            phoneme_segments = []
            
            for i in range(phoneme_count):
                start_frame = i * frames_per_phoneme
                end_frame = (i + 1) * frames_per_phoneme - 1
                phoneme_segments.append((start_frame, end_frame))
            
            example["phoneme_segments"] = phoneme_segments
            return example
            
        except Exception as e:
            print(f"處理 {example.get('uttid', 'unknown')} 時發生錯誤: {e}")
            # 返回預設分段
            phoneme_count = len(example.get("cmu_ipa_phonetic_transcription", []))
            example["phoneme_segments"] = [(0, 10)] * phoneme_count
            return example
    
    # 為資料集添加 phoneme_segments
    print("生成簡單的 phoneme_segments...")
    dataset = dataset.map(add_simple_phoneme_segments)
    
    # 保存更新後的資料集
    print(f"保存到: {output_path}")
    dataset.save_to_disk(output_path)
    
    print("完成！")

if __name__ == "__main__":
    _dataset_path = "/Users/xrickliao/WorkSpaces/DataSets/speechocean762_hf"
    _output_path = "/Users/xrickliao/WorkSpaces/DataSets/fa_phoneme_segment"
    
    generate_simple_phoneme_segments(_dataset_path, _output_path)