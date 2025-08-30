import torch
import os
from torch.utils.data import IterableDataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset
import wandb
from model_smol2 import CustomConfig, CustomLLM

# Configuration - Match exactly with your model specs
CHECKPOINT_DIR = "./checkpoints"
SEQ_LENGTH = 512  # Reduced from 2048 due to memory constraints
BATCH_SIZE = 4    # Adjust based on available memory
GRAD_ACCUM_STEPS = 8  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="no",  # MPS doesn't support FP16
    gradient_accumulation_steps=GRAD_ACCUM_STEPS
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token  # For padding

# Model configuration (from your specs)
config = CustomConfig()
# Align config with tokenizer's special tokens
config.eos_token_id = tokenizer.eos_token_id
config.pad_token_id = tokenizer.pad_token_id
config.bos_token_id = tokenizer.bos_token_id


# Initialize model
model = CustomLLM(config)
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params/1e6:.2f}M")
print(model)
model.to(device)
# Dataset setup (streaming)
class StreamDataset(IterableDataset):
    def __init__(self, split=None, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            # Load dataset with correct configuration
            self.dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                name="cosmopedia-v2",  # Explicitly specify dataset name
                split=split,
                streaming=True
            ).map(
                tokenize_fn,
                batched=True
            )
    
    def __iter__(self):
        for sample in self.dataset:
            yield sample  # Now yields tokenized data

    def take(self, n):
        return StreamDataset(dataset=self.dataset.take(n))


# Tokenization function
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        max_length=SEQ_LENGTH,
        truncation=True,
        padding="max_length"
    )

# Data collator (handles padding and attention masks)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    max_steps=5000, #testing 50 orig 5000
    logging_steps=100, #testing 10 orig 100
    save_steps=500, # orig is 500 
    eval_strategy="steps",
    eval_steps=500, # orig 500
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500, # orig 500
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    max_grad_norm=1.0,
    fp16=False,  # Disabled for MPS
    remove_unused_columns=True,
    report_to="wandb",
    ddp_find_unused_parameters=False,
    save_safetensors=False,
)

# Custom callback for MPS-specific handling
class MPSCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

# Add this new callback class
class TextGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, config, prompt="Hello, AI world!", generation_length=100, temperature=0.7 ):
        self.tokenizer = tokenizer
        self.config = config
        self.prompt = prompt
        self.generation_length = generation_length
        self.temperature = temperature
        
    def on_step_end(self, args, state, control, **kwargs):
        # orig 500
        if state.global_step % 500 == 0 and state.global_step > 0:
            model = kwargs['model']
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                self.prompt, 
                return_tensors="pt",
                padding=False,  # Explicitly disable padding
                return_attention_mask=False
            ).to(device)
            
            # Generate text
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=self.generation_length,
                    temperature=self.temperature,
                    top_p=0.9,
                    eos_token_id=None, ## Disable EOS early stopping
                    pad_token_id=self.config.pad_token_id
                )
            model.train()
            
            # Decode and log
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\nGenerated text at step {state.global_step}:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)
            
            
            wandb.log({"generated_text": wandb.Html(f"<pre>{generated_text}</pre>")}, 
                        step=state.global_step)

# the Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=StreamDataset("train"),
    eval_dataset=StreamDataset("train").take(100),
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[
        MPSCallback(),
        TextGenerationCallback(
            tokenizer,
            config=config,  # Pass the config here
            prompt="The future of AI is",
            generation_length=100,
            temperature=0.3  # Reduce randomness
        ),
    ]
)

# Check if checkpoint exists
checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-5000"
if os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}, resuming training... logging interval changed to 10")
    # Temporarily set max_steps to 50 for the resumed training
    trainer.args.max_steps = 5050
    trainer.args.logging_steps=10
    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model(f"{CHECKPOINT_DIR}/final_5050")
else:
    print(f"No checkpoint found, starting training from scratch...")
    trainer.train()
    trainer.save_model(f"{CHECKPOINT_DIR}/final_5000")

accelerator.print("Training complete!")
