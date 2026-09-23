import pandas as pd
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def main():
    print("Loading data for finetuning...")
    df = pd.read_csv("../operator_sessions.csv")
    
    # Take 2000 rows as requested
    df_sample = df.sample(n=2000, random_state=42)
    
    # Format the data
    formatted_data = []
    for _, row in df_sample.iterrows():
        input_text = f"RPM={int(row['RPM'])}, IdleTime={int(row['IdlingTime'])}min, Seatbelt={row['SeatbeltStatus']}, Speed={row['SpeedKPH']:.1f}kph, HydPressure={int(row['HydraulicPressure'])}bar"
        label = row['AlertType']
        # Convert to Unsloth prompt format
        text = f"Analyze the telemetry: {input_text}\nOutput: {label}"
        formatted_data.append({"text": text})
        
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    print("Initializing Unsloth Model (Qwen2.5:1.5b)...")
    max_seq_length = 512
    dtype = None
    load_in_4bit = True
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-1.5B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "v_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 42,
        use_rslora = False,
        loftq_config = None,
    )
    
    print("Starting training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
        ),
    )
    
    trainer.train()
    
    print("Saving Finetuned LoRA...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("Finetuning complete! Model saved to models/lora_model/")

if __name__ == "__main__":
    main()
