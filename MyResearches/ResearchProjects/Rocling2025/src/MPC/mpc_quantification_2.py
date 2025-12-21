#!/usr/bin/env python3
"""
Goodness of Pronunciation (GOP) Calculation Pipeline
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import datasets
from tqdm import tqdm
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_from_disk

import os
import socket
from datetime import datetime

# Configuration Constants
MODEL_PATH = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
CACHE_DIR = Path("/vol/tensusers6/aparikh/PhD/CTC-based-GOP/cache_dir")
OUTPUT_CSV = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/quantification/mpc_evaluation/forced_aligned_mpc4.csv"

def print_system_info():
    """
    Prints detailed system information including:
    - Start date and time
    - Hostname
    - Machine architecture
    """
    # Get the current date and time
    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get hostname details
    full_hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    os_details = os.uname()
    machine_architecture = os_details.machine

    print(f"Full hostname: {full_hostname}")
    print(f"FQDN: {fqdn}")
    print(f"Machine Architecture: {machine_architecture}")

    # Return start_time for potential runtime calculations
    return start_time


def initialize_components() -> Tuple[Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC]:
    """Initialize model components with proper caching."""
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return processor, tokenizer, model



def process_utterance(
    row: dict,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    tokenizer: Wav2Vec2CTCTokenizer,
    alpha: float = 0.7
) -> List[tuple]:
    """Calculate pronunciation metrics directly from segments without CTC segmentation."""
    results = []
    
    # Audio processing
    audio = row["audio"]["array"]
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    input_values = inputs.input_values.to(model.device)  # Changed to model.device
    
    # Model inference
    with torch.no_grad():
        logits = model(input_values).logits.cpu()[0]
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Process each segment
    for pid_target, pid_actual, start_ms, end_ms in row["segment"]:
        if pid_target == 0:  # Skip padding tokens
            continue
            
        # Time index adjustment (preserving your //2 division)
        start_frame = start_ms // 2
        end_frame = end_ms // 2
        
        # Validate frame indices
        if (start_frame >= end_frame) or (end_frame > logits.shape[0]):
            continue

        # Initialize metrics dict
        metrics = {}  # Added initialization
        
        # Get target token information
        logits_segment = logits[start_frame:end_frame+1]
        target_logits = logits_segment[:, pid_target]
        
        # Calculate metrics
        metrics["max_logit"] = target_logits.max().item()
        metrics["logit_variance"] = target_logits.var(unbiased=False).item()
        
        # Margin calculation
        other_logits = logits_segment.clone()
        other_logits[:, pid_target] = -torch.inf
        max_competitor = other_logits.max(dim=-1).values
        margins = target_logits - max_competitor
        metrics["mean_logit_margin"] = margins.mean().item()

        # Probability calculations
        probs_segment = probs[start_frame:end_frame+1, pid_target]
        metrics["posterior_probability"] = probs_segment.mean().item()  # Fixed typo in key name
        gop_score = -torch.log(torch.tensor(metrics["posterior_probability"]) + 1e-15).item()
        metrics["combined_score"] = (alpha * metrics["mean_logit_margin"] 
                                    - (1-alpha) * gop_score)

        # Phoneme conversion
        target_phone = tokenizer.convert_ids_to_tokens([pid_target])[0]
        actual_phone = tokenizer.convert_ids_to_tokens([pid_actual])[0]

        results.append((
            row["unique_id"],
            target_phone,
            actual_phone,
            start_ms // 1000,  # Start time in seconds
            end_ms // 1000,    # End time in seconds
            metrics["posterior_probability"],  # Explicitly access each metric
            metrics["max_logit"],
            metrics["mean_logit_margin"],
            metrics["logit_variance"],
            metrics["combined_score"],
            int(pid_target != pid_actual)
        ))
    
    return results

def main():
    """Main execution pipeline."""
    # Initialize components
    start_time = print_system_info()
    processor, tokenizer, model = initialize_components()
    
    # Load dataset
    dataset = load_from_disk("/vol/tensusers6/aparikh/PhD/data/mpc/forced_aligned_mpc")
    
    # Process all utterances
    all_results = []
    for row in tqdm(dataset, desc="Processing utterances"):
        all_results.extend(
            process_utterance(
                row=row,
                processor=processor,
                model=model,
                tokenizer=tokenizer,
                alpha=0.7  # Set combination weight
            )
        )
    
    # Save results with updated headers
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uttid", 
            "target_phone", 
            "actual_phone",
            "start_sec",
            "end_sec",
            "posterior_prob",
            "max_logit",
            "mean_margin",
            "logit_variance",
            "combined_score",
            "mispronounced"
        ])
        writer.writerows(all_results)
    
    print(f"\nProcessed {len(all_results)} phoneme entries")
    print(f"Results saved to {OUTPUT_CSV}")
    
    # Print runtime
    end_time = datetime.now()
    print(f"Script finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main()