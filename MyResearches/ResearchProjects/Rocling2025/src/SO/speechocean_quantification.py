import logging
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from typing import List, Dict, Any
import ctc_segmentation
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
from tqdm import tqdm
import csv
import os
import socket
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("ctc_segmentation").setLevel(logging.WARNING)

def print_system_info():
    """
    Prints detailed system information including:
    - Start date and time
    - Hostname
    - Machine architecture
    """
    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    full_hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    os_details = os.uname()
    machine_architecture = os_details.machine

    print(f"Full hostname: {full_hostname}")
    print(f"FQDN: {fqdn}")
    print(f"Machine Architecture: {machine_architecture}")
    return start_time

def extract_phoneme_accuracies(example):
    return {
        'phoneme_accuracies': [
            acc for word in example['words'] for acc in word['phones-accuracy']
        ]
    }
    
def extract_phoneme_accuracies(example):
    """安全地提取音素準確度，處理缺失欄位的情況"""
    # 檢查是否有 words 欄位
    if 'words' in example and example['words'] is not None:
        return {
            'phoneme_accuracies': [
                acc for word in example['words'] for acc in word['phones-accuracy']
            ]
        }
    else:
        # 如果沒有 words 欄位，返回 None 或預設值
        # 根據音素序列長度生成預設值
        phoneme_count = len(example.get('cmu_ipa_phonetic_transcription', []))
        return {
            'phoneme_accuracies': [None] * phoneme_count  # 或使用預設值如 [1.0] * phoneme_count
        }
# def initialize_model(prep_path: str, cache_dir: str, device: torch.device):
#     """
#     Initializes the Wav2Vec2 processor, tokenizer, and model, and moves the model to the specified device.

#     Args:
#         prep_path (str): The identifier or path of the pretrained model.
#         cache_dir (str): The directory to cache the model files.
#         device (torch.device): The device to run the model on.

#     Returns:
#         tuple: A tuple containing the processor, tokenizer, and model.
#     """
#     logger.info("Initializing model components...")
#     processor = Wav2Vec2Processor.from_pretrained(prep_path, cache_dir=cache_dir)
#     tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path, cache_dir=cache_dir)
#     model = Wav2Vec2ForCTC.from_pretrained(prep_path, cache_dir=cache_dir)
#     model.to(device)
#     logger.info("Model initialized and moved to device: %s", device)
#     return processor, tokenizer, model

def initialize_model(prep_path: str, cache_dir: str, device: torch.device):
    """
    Initializes the Wav2Vec2 processor, tokenizer, and model, and moves the model to the specified device.
    """
    logger.info("Initializing model components...")
    
    # 分別初始化各個組件
    # processor = Wav2Vec2Processor.from_pretrained(prep_path, cache_dir=cache_dir)
    # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path, cache_dir=cache_dir)
    # model = Wav2Vec2ForCTC.from_pretrained(prep_path, cache_dir=cache_dir)    
    # # 將模型移動到指定設備
    # model.to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(prep_path, cache_dir=cache_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path, cache_dir=cache_dir)
    model = Wav2Vec2ForCTC.from_pretrained(prep_path, cache_dir=cache_dir)
    model.to(device)
    logger.info("Model initialized and moved to device: %s", device)
    return processor, tokenizer, model

def align_phonemes_with_ctc_frames(
    audio: np.ndarray,
    phoneme_sequence: List[str],
    processor,
    tokenizer,
    model,
    samplerate: int,
    device: torch.device,
    alpha: float = 0.7  # Added alpha parameter for combined score
    ) -> List[Dict[str, Any]]:
    """Enhanced alignment with additional logit-based metrics
    
    Aligns phonemes with CTC frames for a given audio signal and calculates the 
    mean probability (prosetrior_probability) of each phoneme within its aligned frames.

    Args:
        audio (np.ndarray): A 1D numpy array containing mono audio data.
        phoneme_sequence (List[str]): List of phonemes as strings.
        processor: The Wav2Vec2 processor for audio preprocessing.
        tokenizer: The Wav2Vec2 tokenizer for converting phonemes to token IDs.
        model: The Wav2Vec2 model for obtaining logits.
        samplerate (int): The sampling rate of the audio.
        device (torch.device): The device to run computations on.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - "phoneme": The phoneme.
            - "start_time": The start time of the phoneme alignment (in seconds).
            - "end_time": The end time of the phoneme alignment (in seconds).
            - "conf": The original confidence score from segmentation.
            - "prosetrior_probability": The mean probability of the phoneme over its frames.
    """
    assert audio.ndim == 1, "Audio must be mono (1D)."

    inputs = processor(audio, return_tensors="pt", sampling_rate=samplerate, padding="longest")
    inputs.input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

    num_frames = probs.shape[0]
    audio_duration = audio.shape[0] / samplerate
    index_duration = audio_duration / num_frames 

    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

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

    results = []
    for p, s in zip(phoneme_sequence, segments):
        start_time = s[0]
        end_time = s[1]
        start_frame = int(start_time / index_duration)
        end_frame = int(end_time / index_duration)
        start_frame = max(0, start_frame)
        end_frame = min(num_frames - 1, end_frame)
        target_token_id = vocab.get(p)
        metrics = {
            "max_logit": 0.0,
            "mean_logit_margin": 0.0,
            "logit_variance": 0.0,
            "combined_score": 0.0,
            "prosetrior_probability": 0.0
        }

        if target_token_id is not None and start_frame <= end_frame:
            # Get logit segment for target phoneme
            logits_segment = logits[start_frame:end_frame+1]
            target_logits = logits_segment[:, target_token_id]
            
            # Calculate base metrics
            metrics["max_logit"] = target_logits.max().item()
            metrics["logit_variance"] = target_logits.var(unbiased=False).item()
            
            # Calculate margin metrics
            other_logits = logits_segment.clone()
            other_logits[:, target_token_id] = -torch.inf
            max_competitor = other_logits.max(dim=-1).values
            margins = target_logits - max_competitor
            metrics["mean_logit_margin"] = margins.mean().item()
            
            # Calculate probabilities and combined score
            probs_segment = probs[start_frame:end_frame+1, target_token_id]
            metrics["prosetrior_probability"] = float(np.mean(probs_segment))
            gop_score = -np.log(metrics["prosetrior_probability"] + 1e-15)
            metrics["combined_score"] = (alpha * metrics["mean_logit_margin"] 
                                       - (1-alpha) * gop_score)

        results.append({
            "phoneme": p,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "conf": round(s[2], 3),
            **metrics
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
#     ):
#     """
#     Processes the dataset and writes the phoneme alignment details to a CSV file.
#     The CSV file is written in an incremental fashion, flushing after processing each example.
#     Each row now includes the prosetrior_probability (calculated mean probability) and
#     the human rater's phoneme accuracy score.

#     Args:
#         data: The dataset loaded from disk.
#         processor: The Wav2Vec2 processor.
#         tokenizer: The Wav2Vec2 tokenizer.
#         model: The Wav2Vec2 model.
#         samplerate (int): The sampling rate of the audio.
#         csv_output (str): The output CSV file path.
#         device (torch.device): The device to run computations on.
#     """
    
#     with open(csv_output, mode="w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         header = [
#         "uttid", "actual_phoneme", "mispronounced_phoneme",
#         "start_time", "end_time", "confidence",
#         "prosetrior_probability", "max_logit",
#         "mean_logit_margin", "logit_variance",
#         "combined_score", "phoneme_accuracy", "mispronounced"
#         ]
#         writer.writerow(header)

#         for example in tqdm(data, desc="Processing examples"):
#             uttid = example["uttid"]
#             actual_phonemes = example["cmu_ipa_phonetic_transcription"]
#             mispronounced_phonemes = example["cmu_ipa_mispronunciation_transcription"]
#             audio = example["audio"]["array"]
#             human_accuracies = example.get("phoneme_accuracies", [None] * len(actual_phonemes))

#             phoneme_ctc_frames = align_phonemes_with_ctc_frames(
#                 audio, actual_phonemes, processor, tokenizer, model, samplerate, device
#             )

#             for i, (actual, mispronounced) in enumerate(zip(actual_phonemes, mispronounced_phonemes)):
#                 if i < len(phoneme_ctc_frames):
#                     frame_data = phoneme_ctc_frames[i]
#                     human_accuracy = human_accuracies[i] if i < len(human_accuracies) else None
#                     writer.writerow([
#                         uttid,
#                         actual,
#                         mispronounced,
#                         frame_data["start_time"],
#                         frame_data["end_time"],
#                         frame_data["conf"], frame_data["prosetrior_probability"],
#                         frame_data["max_logit"], frame_data["mean_logit_margin"],
#                         frame_data["logit_variance"], frame_data["combined_score"],
#                         human_accuracy, actual != mispronounced  
#                         ])
#             # Flush the file buffer to ensure data is written to disk immediately
#             csvfile.flush()
#     logger.info("CSV saved to %s", csv_output)

def create_csv_data(
    data,
    processor,
    tokenizer,
    model,
    samplerate: int,
    csv_output: str,
    device: torch.device,
    ):
    """
    Processes the dataset and writes the phoneme alignment details to a CSV file.
    """
    
    with open(csv_output, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [
        "uttid", "actual_phoneme", "mispronounced_phoneme",
        "start_time", "end_time", "confidence",
        "prosetrior_probability", "max_logit",
        "mean_logit_margin", "logit_variance",
        "combined_score", "phoneme_accuracy", "mispronounced"
        ]
        writer.writerow(header)

        for example in tqdm(data, desc="Processing examples"):
            uttid = example["uttid"]
            actual_phonemes = example["cmu_ipa_phonetic_transcription"]
            mispronounced_phonemes = example["cmu_ipa_mispronunciation_transcription"]
            
            # 修改這裡：處理音頻檔案路徑
            audio_path = example["audio"]
            if isinstance(audio_path, str):
                # 如果是檔案路徑，需要載入音頻
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    # 如果是立體聲，轉為單聲道
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    # 重採樣到目標取樣率
                    if sr != samplerate:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=samplerate)
                except ImportError:
                    # 如果沒有 soundfile，嘗試使用 torchaudio
                    try:
                        import torchaudio
                        audio, sr = torchaudio.load(audio_path)
                        audio = audio.numpy().squeeze()
                        if audio.ndim > 1:
                            audio = audio.mean(axis=0)
                        if sr != samplerate:
                            audio = torchaudio.functional.resample(
                                torch.tensor(audio).unsqueeze(0), sr, samplerate
                            ).squeeze().numpy()
                    except ImportError:
                        logger.error(f"需要安裝 soundfile 或 torchaudio 來處理音頻檔案")
                        continue
            else:
                # 如果已經是音頻陣列（舊格式）
                audio = audio_path["array"] if isinstance(audio_path, dict) else audio_path
            
            human_accuracies = example.get("phoneme_accuracies", [None] * len(actual_phonemes))

            phoneme_ctc_frames = align_phonemes_with_ctc_frames(
                audio, actual_phonemes, processor, tokenizer, model, samplerate, device
            )

            for i, (actual, mispronounced) in enumerate(zip(actual_phonemes, mispronounced_phonemes)):
                if i < len(phoneme_ctc_frames):
                    frame_data = phoneme_ctc_frames[i]
                    human_accuracy = human_accuracies[i] if i < len(human_accuracies) else None
                    writer.writerow([
                        uttid,
                        actual,
                        mispronounced,
                        frame_data["start_time"],
                        frame_data["end_time"],
                        frame_data["conf"], frame_data["prosetrior_probability"],
                        frame_data["max_logit"], frame_data["mean_logit_margin"],
                        frame_data["logit_variance"], frame_data["combined_score"],
                        human_accuracy, actual != mispronounced  
                        ])
            # Flush the file buffer to ensure data is written to disk immediately
            csvfile.flush()
    logger.info("CSV saved to %s", csv_output)

if __name__ == "__main__":
    SAMPLERATE = 16000
    # PREP_PATH = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    # DS_CACHE_PATH = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/cache_dir"
    # DATA_PATH = "/vol/tensusers6/aparikh/PhD/data/spo762/so_everything_cmu_ipa"
    # CSV_OUTPUT = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/quantification/speechocean_evaluation/phoneme_alignment_CTC_SEGMENT_mean.csv"
    PREP_PATH = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    DS_CACHE_PATH = "../cache_dir"
    # DATA_PATH = "/Users/xrickliao/WorkSpaces/DataSets/speechocean762_hf" #"../../../../../DataSets/speechocean762/speechocean762/"
    DATA_PATH = "/Users/xrickliao/WorkSpaces/DataSets/speechocean762_hf_no_audio" #"../../../../../DataSets/speechocean762/speechocean762/"
    CSV_OUTPUT = "../output/phoneme_alignment_CTC_SEGMENT_mean.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = print_system_info()
    logger.info("Using device: %s", device)
    processor, tokenizer, model = initialize_model(PREP_PATH, DS_CACHE_PATH, device)
    logger.info("Loading dataset from %s", DATA_PATH)
    so_data = load_from_disk(DATA_PATH)
    #so_data = so_data.shuffle(seed=50).select(range(2))
    so_data = so_data.map(extract_phoneme_accuracies)
    create_csv_data(so_data, processor, tokenizer, model, SAMPLERATE, CSV_OUTPUT, device)