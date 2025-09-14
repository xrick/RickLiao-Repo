# generate_phoneme_segments.py
import os
import glob
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk
import argparse
import ctc_segmentation
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2ForCTC

def generate_phoneme_segments_for_dataset(dataset_path: str, output_path: str, model_path: str):
    """為資料集生成 phoneme_segments 欄位"""
    
    # 載入資料集
    print(f"載入資料集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # 初始化模型
    print("初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    
    def add_phoneme_segments(example):
        """為單個樣本生成 phoneme_segments"""
        try:
            # 處理音頻
            audio_path = example["audio"]
            if isinstance(audio_path, str):
                # 載入音頻檔案
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio = audio_path["array"]
            
            # 獲取 logits
            inputs = processor(audio, return_tensors="pt", sampling_rate=16000, padding="longest")
            inputs.input_values = inputs.input_values.to(device)
            
            with torch.no_grad():
                logits = model(inputs.input_values).logits.cpu()[0]
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
            
            # 使用 CTC 分割生成對齊
            num_frames = probs.shape[0]
            audio_duration = len(audio) / 16000
            index_duration = audio_duration / num_frames
            
            vocab = tokenizer.get_vocab()
            inv_vocab = {v: k for k, v in vocab.items()}
            
            phoneme_sequence = example["cmu_ipa_phonetic_transcription"]
            tokenized_phonemes = []
            for phoneme in phoneme_sequence:
                token_ids = tokenizer(phoneme)["input_ids"]
                token_ids = np.array(token_ids, dtype=np.int32)
                tokenized = token_ids[token_ids != vocab.get("[UNK]", -1)]
                tokenized_phonemes.append(tokenized)
            
            char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
            config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
            config.index_duration = index_duration
            
            ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokenized_phonemes)
            timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs, ground_truth_mat)
            segments = ctc_segmentation.determine_utterance_segments(
                config, utt_begin_indices, char_probs, timings, phoneme_sequence
            )
            
            # 轉換為幀級分段
            phoneme_segments = []
            for start_time, end_time, conf in segments:
                start_frame = int(start_time / index_duration)
                end_frame = int(end_time / index_duration)
                start_frame = max(0, start_frame)
                end_frame = min(num_frames - 1, end_frame)
                phoneme_segments.append((start_frame, end_frame))
            
            example["phoneme_segments"] = phoneme_segments
            return example
            
        except Exception as e:
            print(f"處理 {example.get('uttid', 'unknown')} 時發生錯誤: {e}")
            # 返回預設分段
            phoneme_count = len(example.get("cmu_ipa_phonetic_transcription", []))
            example["phoneme_segments"] = [(0, 10)] * phoneme_count  # 預設分段
            return example
    
    # 為資料集添加 phoneme_segments
    print("生成 phoneme_segments...")
    dataset = dataset.map(add_phoneme_segments)
    
    # 保存更新後的資料集
    print(f"保存到: {output_path}")
    dataset.save_to_disk(output_path)
    
    print("完成！")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_path", default="/Users/xrickliao/WorkSpaces/DataSets/speechocean762_hf", required=True, help="原始資料集路徑")
    # parser.add_argument("--output_path", default="/Users/xrickliao/WorkSpaces/DataSets/fa_phoneme_segment", required=True, help="輸出資料集路徑")
    # parser.add_argument("--model_path", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft", help="模型路徑")
    
    # args = parser.parse_args()
    _dataset_path = "/Users/xrickliao/WorkSpaces/DataSets/speechocean762_hf"
    _output_path = "/Users/xrickliao/WorkSpaces/DataSets/fa_phoneme_segment"
    _model_path = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    generate_phoneme_segments_for_dataset(_dataset_path, _output_path, _model_path)
    # generate_phoneme_segments_for_dataset(args.dataset_path, args.output_path, args.model_path)