# train.py

# -----------------------------
# MUST set these BEFORE any imports from transformers/trl
import os
os.environ["TRANSFORMERS_NO_TRACKIO"] = "1"  # disables internal tracking
os.environ["TRL_GRADIO_ENABLED"] = "0"      # disables Gradio UI
os.environ["WANDB_MODE"] = "offline"        # optional: disables WandB logging

# -----------------------------
import torch
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, TrainingArguments 
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# -----------------------------

model_name = "openai/gpt-oss-20b"
tokenizer_name = model_name

quantization_config = Mxfp4Config(dequantize=True)

model_kwargs = dict(
    attn_implementation='eager', # better performance
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False, 
    device_map='auto') 

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

# -----------------------------

peft_config = LoraConfig(
    r=8,                   # enable more trainable parameters
    lora_alpha=16,
    target_modules='all-linear', # layers to target ie: attn
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj"
    ]
)

peft_model = get_peft_model(model,peft_config)
# -----------------------------

filepath = '/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/Files'

training_args = TrainingArguments(    # SFTConfig
    output_dir=os.path.join(filepath,"gpt-oss-20b-multilingual-reasoner"),
    save_strategy="steps",
    save_steps=50,
    fp16=True,  # enable if using GPU
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # accumulate gradients over multiple batches before weight update 
    # max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate":0.1},
    report_to="none",
)
# -----------------------------

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
# -----------------------------

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,#.shuffle(buffer_size=1000),
    processing_class=tokenizer   
)
# -----------------------------

trainer.train()
# -----------------------------
