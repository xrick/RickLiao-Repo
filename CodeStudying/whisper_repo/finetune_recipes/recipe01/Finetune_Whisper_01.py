"""
重要參數說明
LoRA配置參數:

r: LoRA的秩，決定了適應層的複雜度
lora_alpha: 縮放因子，影響LoRA更新的強度
lora_dropout: 用於防止過擬合
target_modules: 指定要應用LoRA的層
訓練參數:

learning_rate: 建議使用較小的學習率（1e-3 到 1e-4）
batch_size: 根據GPU記憶體大小調整
gradient_accumulation_steps: 用於處理較大批次
"""
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# 1. 載入基礎模型和處理器
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# 2. 定義LoRA配置
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,                # LoRA的秩
    lora_alpha=32,      # LoRA的縮放因子
    lora_dropout=0.1,   # Dropout率
    target_modules=["q_proj", "v_proj"]  # 要微調的層
)

# 3. 將模型轉換為LoRA模型
model = get_peft_model(model, peft_config)

# 4. 準備訓練數據
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"], 
        return_tensors="pt"
    ).input_features[0]
    
    batch["labels"] = processor(text=batch["text"]).input_ids
    return batch

# 5. 載入數據集（這裡使用Common Voice作為示例）
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-TW", split="train[:100]")
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# 6. 設定訓練參數
training_args = {
    "learning_rate": 1e-3,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
}

# 7. 訓練循環
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args["learning_rate"])

model.train()
for epoch in range(training_args["num_train_epochs"]):
    for i in range(0, len(dataset), training_args["per_device_train_batch_size"]):
        batch = dataset[i:i + training_args["per_device_train_batch_size"]]
        
        input_features = torch.stack(batch["input_features"])
        labels = torch.tensor(batch["labels"])
        
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        if (i + 1) % training_args["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")

# 8. 保存模型
model.save_pretrained("whisper-small-zh-lora")
