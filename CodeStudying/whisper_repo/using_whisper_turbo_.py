# Code used for Turbo model:
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import random
import os
import argparse
import warnings
import sys

# Ignore warnings
warnings.filterwarnings("ignore")

# Redirect standard error to suppress error messages
sys.stderr = open(os.devnull, 'w')

# ASR model
model_id = "ylacombe/whisper-large-v3-turbo"

# Apple Silicon

# Check if MPS is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS device is not available.")

device = "mps"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model = model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device=device,
    return_timestamps=True,
)

def transcribe_mp3(mp3_path):
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Invalid path. {mp3_path}")

    result = pipe(mp3_path)
    output_dir = os.path.expanduser("~/Documents/transcription-texts")
    os.makedirs(output_dir, exist_ok=True)
    random_number = str(random.randint(10, 99))
    output_filename = f"transcription_{random_number}.txt"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        f.write(result["text"])
    print(f"Transcription saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe MP3 file.")
    parser.add_argument("mp3_path", help="Path to audio file")
    args = parser.parse_args()

    try:
        transcribe_mp3(args.mp3_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")