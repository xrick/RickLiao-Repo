# 載入微調後的模型
from peft import PeftModel, PeftConfig

# 載入基礎模型
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# 載入LoRA權重
model = PeftModel.from_pretrained(base_model, "whisper-small-zh-lora")

# 進行推論
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def transcribe_audio(audio_path):
    # 載入音頻
    audio_input = processor(
        audio_path, 
        return_tensors="pt"
    ).input_features
    
    # 生成轉錄
    predicted_ids = model.generate(input_features=audio_input)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]
