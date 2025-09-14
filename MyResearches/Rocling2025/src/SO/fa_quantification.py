import logging
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from typing import List, Dict, Any, Tuple
import ctc_segmentation
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
from tqdm import tqdm
import csv
import os
import socket
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_system_info():
    """Print system information and return start time."""
    from time import time, ctime
    start_time = time()
    logger.info(f"System time: {ctime(start_time)}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    return start_time

def initialize_model(model_path, cache_dir, device):
    """Initialize and return processor, tokenizer, and model."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
    
    logger.info("Initializing model components...")
    
    # 使用 FeatureExtractor 替代 Processor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, cache_dir=cache_dir)
    logger.info(f"Processor 類型: {type(processor)}")
    
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    logger.info(f"Tokenizer 類型: {type(tokenizer)}")
    logger.info(f"Tokenizer 是否有 get_vocab: {hasattr(tokenizer, 'get_vocab')}")
    
    model = Wav2Vec2ForCTC.from_pretrained(model_path, cache_dir=cache_dir)
    logger.info(f"Model 類型: {type(model)}")
    
    # 將模型移動到指定設備
    model.to(device)
    
    logger.info("Model initialized and moved to device: %s", device)
    return processor, tokenizer, model

def softmax(x):
    """Compute softmax with numerical stability."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_temp(x, T=1.0):
    """Compute temperature-scaled softmax."""
    x = x / T
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def logit_regularization(logits):
    """Apply L2 normalization per frame."""
    norms = np.linalg.norm(logits, axis=-1, keepdims=True)
    return logits / np.where(norms > 0, norms, 1.0)


def entropy(p, axis=-1):
    """Calculate Shannon entropy."""
    return -np.sum(p * np.log(p + 1e-10), axis=axis)

def renyi_entropy(p, alpha=2.0, axis=-1):
    """Compute Rényi entropy."""
    if alpha == 1:
        return entropy(p, axis=axis)
    return (1 / (1 - alpha)) * np.log(np.sum(p**alpha, axis=axis) + 1e-10)

def tsallis_entropy(p, alpha=2.0, axis=-1):
    """Compute Tsallis entropy."""
    if alpha == 1:
        return entropy(p, axis=axis)
    return (1 / (alpha - 1)) * (np.sum(p**alpha, axis=axis) - 1)

def aggregate_values(values, method="mean"):
    """Aggregate values using the specified method."""
    if method == "mean":
        return np.mean(values)
    elif method == "median":
        return np.median(values)
    elif method == "min":
        return np.min(values)
    elif method == "max":
        return np.max(values)
    elif method == "prod":
        return np.prod(values)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

# def align_phonemes_with_ctc_frames(
#     audio: np.ndarray,
#     phoneme_sequence: List[str],
#     phoneme_segments: List[Tuple[int, int]],  # Now accepted!
#     processor,
#     tokenizer,
#     model,
#     samplerate: int,
#     device: torch.device,
#     temperature: float = 1.0,
#     aggregation_method: str = "mean"
# ) -> List[Dict[str, Any]]:
#     """
#     Aligns phonemes using provided segmentation (assumed to be frame indices) for a given audio signal
#     and calculates various metrics for each phoneme, including maximum probability, temperature-scaled probability,
#     max logit, logit margin, and entropy measures over the segment.

#     Args:
#         audio (np.ndarray): A 1D numpy array containing mono audio data.
#         phoneme_sequence (List[str]): List of phonemes as strings.
#         phoneme_segments (List[Tuple[int, int]]): List of tuples (start_frame, end_frame) for each phoneme.
#         processor: The Wav2Vec2 processor for audio preprocessing.
#         tokenizer: The Wav2Vec2 tokenizer for converting phonemes to token IDs.
#         model: The Wav2Vec2 model for obtaining logits.
#         samplerate (int): The sampling rate of the audio.
#         device (torch.device): The device to run computations on.
#         temperature (float): Temperature for softmax scaling.
#         aggregation_method (str): Aggregation method for entropy measures.

#     Returns:
#         List[Dict[str, Any]]: A list of dictionaries, each containing:
#             - "phoneme": The phoneme.
#             - "start_frame": The start frame of the phoneme.
#             - "end_frame": The end frame of the phoneme.
#             - "conf": Default confidence (set to 1.0).
#             - "posterior_prob_standard": The maximum probability for the phoneme over its frames.
#             - "posterior_prob_temp": The maximum temperature-scaled softmax probability.
#             - "max_logit": The maximum logit for the target token in the segment.
#             - "logit_margin": The difference between the max logit and the second highest logit 
#                               in the frame with the max logit.
#             - "entropy": Aggregated Shannon entropy over the segment.
#             - "renyi_entropy": Aggregated Rényi entropy over the segment.
#             - "tsallis_entropy": Aggregated Tsallis entropy over the segment.
#     """
#     assert audio.ndim == 1, "Audio must be mono (1D)."
    
#     # Preprocess audio and move inputs to device.
#     inputs = processor(audio, return_tensors="pt", sampling_rate=samplerate, padding="longest")
#     inputs.input_values = inputs.input_values.to(device)
    
#     with torch.no_grad():
#         logits = model(inputs.input_values).logits.cpu()[0]  # shape: [num_frames, vocab_size]
#         probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
#     print("Logits shape:", logits.shape)
    
#     # Compute temperature-scaled probabilities.
#     probs_standard = softmax_temp(logits.numpy(), T=temperature)
    
#     vocab = tokenizer.get_vocab()
#     inv_vocab = {v: k for k, v in vocab.items()}
    
#     results = []
#     # Use provided phoneme_segments directly.
#     segments = phoneme_segments
#     for p, s in zip(phoneme_sequence, segments):
#         # Here, s is assumed to be a tuple (start_frame, end_frame)
#         start_frame = s[0]
#         end_frame = s[1]
#         conf = 1.0  # default confidence
        
#         target_token_id = vocab.get(p)
        
#         if target_token_id is None:
#             mean_prob = 0.0
#             softmax_prob = 0.0
#             maxlogits = 0.0
#             logit_margin = 0.0
#             aggregated_entropy = 0.0
#             aggregated_renyi = 0.0
#             aggregated_tsallis = 0.0
#         else:
#             # Calculate maximum probability values within the segment for the target token.
#             mean_prob = float(np.max(probs[start_frame:end_frame + 1, target_token_id]))
#             softmax_prob = float(np.max(probs_standard[start_frame:end_frame + 1, target_token_id]))
            
#             # Compute max logit directly.
#             maxlogits = float(np.max(logits[start_frame:end_frame + 1, target_token_id].numpy()))
            
#             # Compute logit margin for the frame where target token's logit is maximum.
#             seg_slice = logits[start_frame:end_frame + 1, target_token_id].numpy()
#             frame_idx = np.argmax(seg_slice)
#             frame_logits = logits[start_frame:end_frame + 1].numpy()[frame_idx]
#             sorted_logits = np.sort(frame_logits)[::-1]
#             logit_margin = float(sorted_logits[0] - sorted_logits[1]) if sorted_logits.size > 1 else 0.0
            
#             # Calculate entropy measures over the segment using the entire probability distribution.
#             seg_probs = probs[start_frame:end_frame + 1, :]  # shape: [segment_frames, vocab_size]
#             frame_entropy = entropy(seg_probs, axis=-1)
#             aggregated_entropy = aggregate_values(frame_entropy, method=aggregation_method)
#             frame_renyi = renyi_entropy(seg_probs, alpha=0.33, axis=-1)
#             aggregated_renyi = aggregate_values(frame_renyi, method="max")
#             frame_tsallis = tsallis_entropy(seg_probs, alpha=0.33, axis=-1)
#             aggregated_tsallis = aggregate_values(frame_tsallis, method="max")
            
#         results.append({
#             "phoneme": p,
#             "start_frame": start_frame,
#             "end_frame": end_frame,
#             "conf": round(conf, 3),
#             "posterior_prob_standard": round(mean_prob, 3),
#             "posterior_prob_temp": round(softmax_prob, 3),
#             "max_logit": round(maxlogits, 3),
#             "logit_margin": round(logit_margin, 3),
#             "entropy": round(aggregated_entropy, 3),
#             "renyi_entropy": round(aggregated_renyi, 3),
#             "tsallis_entropy": round(aggregated_tsallis, 3)
#         })
    
#     return results

def align_phonemes_with_ctc_frames(
    audio,  # 現在可能是檔案路徑或音頻陣列
    phoneme_sequence: List[str],
    phoneme_segments: List[Tuple[int, int]],
    processor,
    tokenizer,
    model,
    samplerate: int,
    device: torch.device,
    temperature: float = 1.0,
    aggregation_method: str = "mean"
) -> List[Dict[str, Any]]:
    """
    Aligns phonemes using provided segmentation (assumed to be frame indices) for a given audio signal
    and calculates various metrics for each phoneme.
    """
    # 處理音頻輸入
    if isinstance(audio, str):
        # 如果是檔案路徑，需要先載入音頻
        try:
            import soundfile as sf
            audio_array, sr = sf.read(audio)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)  # 轉為單聲道
            if sr != samplerate:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=samplerate)
        except ImportError:
            try:
                import torchaudio
                audio_array, sr = torchaudio.load(audio)
                audio_array = audio_array.numpy().squeeze()
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=0)
                if sr != samplerate:
                    audio_array = torchaudio.functional.resample(
                        torch.tensor(audio_array).unsqueeze(0), sr, samplerate
                    ).squeeze().numpy()
            except ImportError:
                raise RuntimeError("需要安裝 soundfile 或 torchaudio 來載入音頻檔案")
        
        # 使用載入的音頻陣列
        inputs = processor(audio_array, return_tensors="pt", sampling_rate=samplerate, padding="longest")
    else:
        # 如果是音頻陣列
        assert audio.ndim == 1, "Audio must be mono (1D)."
        inputs = processor(audio, return_tensors="pt", sampling_rate=samplerate, padding="longest")
    
    inputs.input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]  # shape: [num_frames, vocab_size]
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
    print("Logits shape:", logits.shape)
    
    # 其餘代碼保持不變...
    # Compute temperature-scaled probabilities.
    probs_standard = softmax_temp(logits.numpy(), T=temperature)
    
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    
    results = []
    # Use provided phoneme_segments directly.
    segments = phoneme_segments
    for p, s in zip(phoneme_sequence, segments):
        # Here, s is assumed to be a tuple (start_frame, end_frame)
        start_frame = s[0]
        end_frame = s[1]
        conf = 1.0  # default confidence
        
        target_token_id = vocab.get(p)
        
        if target_token_id is None:
            mean_prob = 0.0
            softmax_prob = 0.0
            maxlogits = 0.0
            logit_margin = 0.0
            aggregated_entropy = 0.0
            aggregated_renyi = 0.0
            aggregated_tsallis = 0.0
        else:
            # Calculate maximum probability values within the segment for the target token.
            mean_prob = float(np.max(probs[start_frame:end_frame + 1, target_token_id]))
            softmax_prob = float(np.max(probs_standard[start_frame:end_frame + 1, target_token_id]))
            
            # Compute max logit directly.
            maxlogits = float(np.max(logits[start_frame:end_frame + 1, target_token_id].numpy()))
            
            # Compute logit margin for the frame where target token's logit is maximum.
            seg_slice = logits[start_frame:end_frame + 1, target_token_id].numpy()
            frame_idx = np.argmax(seg_slice)
            frame_logits = logits[start_frame:end_frame + 1].numpy()[frame_idx]
            sorted_logits = np.sort(frame_logits)[::-1]
            logit_margin = float(sorted_logits[0] - sorted_logits[1]) if sorted_logits.size > 1 else 0.0
            
            # Calculate entropy measures over the segment using the entire probability distribution.
            seg_probs = probs[start_frame:end_frame + 1, :]  # shape: [segment_frames, vocab_size]
            frame_entropy = entropy(seg_probs, axis=-1)
            aggregated_entropy = aggregate_values(frame_entropy, method=aggregation_method)
            frame_renyi = renyi_entropy(seg_probs, alpha=0.33, axis=-1)
            aggregated_renyi = aggregate_values(frame_renyi, method="max")
            frame_tsallis = tsallis_entropy(seg_probs, alpha=0.33, axis=-1)
            aggregated_tsallis = aggregate_values(frame_tsallis, method="max")
            
        results.append({
            "phoneme": p,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "conf": round(conf, 3),
            "posterior_prob_standard": round(mean_prob, 3),
            "posterior_prob_temp": round(softmax_prob, 3),
            "max_logit": round(maxlogits, 3),
            "logit_margin": round(logit_margin, 3),
            "entropy": round(aggregated_entropy, 3),
            "renyi_entropy": round(aggregated_renyi, 3),
            "tsallis_entropy": round(aggregated_tsallis, 3)
        })
    
    return results

# def create_csv_data(
#     data,
#     processor,
#     tokenizer,
#     model,
#     samplerate: int,
#     csv_output: str,
#     device: torch.device,
#     temperature: float = 1.0,
#     aggregation_method: str = "mean"
# ):
#     """
#     Process the dataset and write phoneme alignment details (with segmentation) to CSV.
#     Assumes that each example in `data` contains:
#       - "audio": {"array": np.ndarray, "path": str}
#       - "cmu_ipa_phonetic_transcription": list of phonemes
#       - "cmu_ipa_mispronunciation_transcription": list of mispronounced phonemes
#       - "phoneme_segments": list of tuples (start_frame, end_frame) for each phoneme
#     """
#     with open(csv_output, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         header = [
#             "uttid", "actual_phoneme", "mispronounced_phoneme",
#             "start_frame", "end_frame",
#             "posterior_prob_standard", "posterior_prob_temp",
#             "max_logit", "logit_margin", "entropy",
#             "renyi_entropy", "tsallis_entropy", "mispronounced"
#         ]
#         writer.writerow(header)

#         for example in tqdm(data, desc="Processing examples"):
#             try:
#                 uttid = example["uttid"]
#                 actual_phonemes = example["cmu_ipa_phonetic_transcription"]
#                 mispronounced = example["cmu_ipa_mispronunciation_transcription"]
#                 phoneme_segments = example["phoneme_segments"]  # List of (start_frame, end_frame)
#                 audio = example["audio"]["array"]
                
#                 if len(actual_phonemes) != len(mispronounced) or len(actual_phonemes) != len(phoneme_segments):
#                     logger.warning(f"Skipping {uttid} due to mismatch in phoneme lengths or missing segments")
#                     continue

#                 alignments = align_phonemes_with_ctc_frames(
#                     audio, actual_phonemes, phoneme_segments,
#                     processor, tokenizer, model, samplerate,
#                     device, temperature, aggregation_method
#                 )

#                 for actual, mispron, alignment in zip(actual_phonemes, mispronounced, alignments):
#                     writer.writerow([
#                         uttid, actual, mispron,
#                         alignment["start_frame"],
#                         alignment["end_frame"],
#                         alignment["posterior_prob_standard"],
#                         alignment["posterior_prob_temp"],
#                         alignment["max_logit"],
#                         alignment["logit_margin"],
#                         alignment["entropy"],
#                         alignment["renyi_entropy"],
#                         alignment["tsallis_entropy"],
#                         actual != mispron
#                     ])
                
#                 csvfile.flush()
            
#             except Exception as e:
#                 logger.error(f"Error processing {example.get('audio', {}).get('path', 'unknown')}: {str(e)}")

#     logger.info(f"Successfully saved CSV to {csv_output}")

def create_csv_data(
    data,
    processor,
    tokenizer,
    model,
    samplerate: int,
    csv_output: str,
    device: torch.device,
    temperature: float = 1.0,
    aggregation_method: str = "mean"
):
    """
    Process the dataset and write phoneme alignment details to CSV.
    """
    with open(csv_output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "uttid", "actual_phoneme", "mispronounced_phoneme",
            "start_frame", "end_frame",
            "posterior_prob_standard", "posterior_prob_temp",
            "max_logit", "logit_margin", "entropy",
            "renyi_entropy", "tsallis_entropy", "mispronounced"
        ]
        writer.writerow(header)

        for example in tqdm(data, desc="Processing examples"):
            try:
                uttid = example["uttid"]
                actual_phonemes = example["cmu_ipa_phonetic_transcription"]
                mispronounced = example["cmu_ipa_mispronunciation_transcription"]
                
                # 處理音頻輸入
                audio_input = example["audio"]
                if isinstance(audio_input, str):
                    audio = audio_input
                else:
                    audio = audio_input["array"]
                
                # 檢查是否有 phoneme_segments
                if "phoneme_segments" in example:
                    phoneme_segments = example["phoneme_segments"]
                else:
                    # 如果沒有 phoneme_segments，生成簡單的等長分段
                    phoneme_count = len(actual_phonemes)
                    frames_per_phoneme = 10  # 可調整
                    phoneme_segments = []
                    for i in range(phoneme_count):
                        start_frame = i * frames_per_phoneme
                        end_frame = (i + 1) * frames_per_phoneme - 1
                        phoneme_segments.append((start_frame, end_frame))
                
                if len(actual_phonemes) != len(mispronounced) or len(actual_phonemes) != len(phoneme_segments):
                    logger.warning(f"Skipping {uttid} due to mismatch in phoneme lengths or missing segments")
                    continue

                alignments = align_phonemes_with_ctc_frames(
                    audio, actual_phonemes, phoneme_segments,
                    processor, tokenizer, model, samplerate,
                    device, temperature, aggregation_method
                )

                for actual, mispron, alignment in zip(actual_phonemes, mispronounced, alignments):
                    writer.writerow([
                        uttid, actual, mispron,
                        alignment["start_frame"],
                        alignment["end_frame"],
                        alignment["posterior_prob_standard"],
                        alignment["posterior_prob_temp"],
                        alignment["max_logit"],
                        alignment["logit_margin"],
                        alignment["entropy"],
                        alignment["renyi_entropy"],
                        alignment["tsallis_entropy"],
                        actual != mispron
                    ])
                
                csvfile.flush()
            
            except Exception as e:
                # 修正錯誤處理
                audio_info = "unknown"
                if isinstance(example.get("audio"), str):
                    audio_info = example["audio"]
                elif isinstance(example.get("audio"), dict):
                    audio_info = example.get("audio", {}).get("path", "unknown")
                logger.error(f"Error processing {audio_info}: {str(e)}")

    logger.info(f"Successfully saved CSV to {csv_output}")

if __name__ == "__main__":
    SAMPLERATE = 16000
    PREP_PATH = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    # DS_CACHE_PATH = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/cache_dir"
    # DATA_PATH = "/vol/tensusers6/aparikh/PhD/data/mpc/simulated_error_mpc"
    # CSV_OUTPUT = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/quantification/speechocean/qunatified_gop.csv"
    DS_CACHE_PATH = "../cache_dir"
    DATA_PATH = "/Users/xrickliao/WorkSpaces/DataSets/speechocean762_hf_no_audio"
    CSV_OUTPUT = "../output/qunatified_gop.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = print_system_info()
    logger.info("Using device: %s", device)
    processor, tokenizer, model = initialize_model(PREP_PATH, DS_CACHE_PATH, device)
    logger.info("Loading dataset from %s", DATA_PATH)
    so_data = load_from_disk(DATA_PATH)
    so_data = so_data.shuffle(seed=50).select(range(2))
    create_csv_data(so_data, processor, tokenizer, model, SAMPLERATE, CSV_OUTPUT, device)
