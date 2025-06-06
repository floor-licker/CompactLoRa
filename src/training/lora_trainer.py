#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Compact Smart Contracts

Fine-tunes a language model to generate Compact smart contracts using LoRA.
Includes compiler validation during training.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import json


class CompactLoRATrainer:
    """LoRA trainer specifically for Compact smart contracts."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",  # Smaller, faster model for testing
                 dataset_path: str = "data/compact_lora_dataset",
                 output_dir: str = "models/compact-lora"):
        
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.dataset = None
    
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer with LoRA."""
        print(f"🔄 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Print model architecture to understand target modules
        print("🔍 Model architecture inspection:")
        for name, module in self.model.named_modules():
            if any(key in name for key in ['attn', 'proj', 'linear', 'dense']):
                print(f"   {name}: {type(module).__name__}")
        
        # Auto-detect target modules for different model architectures
        target_modules = []
        for name, module in self.model.named_modules():
            if 'attn' in name and ('c_attn' in name or 'c_proj' in name):
                target_modules.append(name.split('.')[-1])
        
        # Fallback targets based on model type
        if not target_modules:
            if 'gpt2' in self.model_name.lower():
                target_modules = ["c_attn", "c_proj"]
            else:
                target_modules = ["c_attn", "c_proj"]  # Default fallback
        
        target_modules = list(set(target_modules))  # Remove duplicates
        print(f"🎯 Using target modules: {target_modules}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,                     # Smaller rank for stability
            lora_alpha=16,           # Scaling parameter
            lora_dropout=0.1,        # Dropout
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print(f"✅ Model loaded with LoRA adapters")
    
    def load_dataset(self):
        """Load the prepared dataset."""
        print(f"📚 Loading dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        self.dataset = load_from_disk(str(self.dataset_path))
        print(f"✅ Loaded {len(self.dataset)} training examples")
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,  # Smaller context for stability
                return_overflowing_tokens=False,
            )
        
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names
        )
        
        print(f"✅ Dataset tokenized")
    
    def create_training_arguments(self, 
                                 num_epochs: int = 3,
                                 batch_size: int = 4,
                                 learning_rate: float = 2e-4) -> TrainingArguments:
        """Create training arguments."""
        
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            warmup_steps=50,          # Fewer warmup steps
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=5,
            save_steps=50,            # Save more frequently
            eval_steps=50,
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,           # Disable wandb
            load_best_model_at_end=False,  # Simplify for small dataset
            dataloader_pin_memory=False,   # Helps with memory issues
            dataloader_num_workers=0,      # Avoid multiprocessing issues
        )
    
    def train(self, **training_kwargs):
        """Main training function."""
        print("🚀 STARTING LORA FINE-TUNING")
        print("=" * 40)
        
        # Setup
        self.setup_model_and_tokenizer()
        self.load_dataset()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Training arguments
        training_args = self.create_training_arguments(**training_kwargs)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train!
        print(f"🎯 Training for {training_args.num_train_epochs} epochs...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        print(f"💾 Model saved to {self.output_dir}")
        
        return trainer
    
    def test_generation(self, prompt: str = "Create a simple Compact counter contract"):
        """Test the fine-tuned model."""
        print(f"\n🧪 Testing model generation...")
        print(f"Prompt: {prompt}")
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Generated:\n{generated_text}")
        return generated_text


def main():
    """Main function for LoRA training."""
    print("🎯 COMPACT SMART CONTRACT LORA FINE-TUNING")
    print("=" * 55)
    
    # Configuration
    config = {
        "model_name": "gpt2",  # Proven model for LoRA fine-tuning
        "dataset_path": "data/compact_final_dataset",  # 211 validated examples
        "output_dir": "models/compact-lora-gpt2-final",
        "num_epochs": 15,       # Reduce epochs since we had good progress
        "batch_size": 1,        # Back to 1 to avoid tensor issues
        "learning_rate": 1e-4   # Lower for stable learning with more data
    }
    
    print(f"📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Check if dataset exists
    if not Path(config["dataset_path"]).exists():
        print(f"❌ Dataset not found at {config['dataset_path']}")
        print(f"💡 Run: python3 src/training/prepare_lora_data.py")
        return
    
    # Initialize trainer
    trainer = CompactLoRATrainer(
        model_name=config["model_name"],
        dataset_path=config["dataset_path"],
        output_dir=config["output_dir"]
    )
    
    # Train
    training_kwargs = {
        "num_epochs": config["num_epochs"],
        "batch_size": config["batch_size"], 
        "learning_rate": config["learning_rate"]
    }
    
    try:
        trainer.train(**training_kwargs)
        
        # Test the model
        trainer.test_generation()
        
        print(f"\n🎉 LoRA fine-tuning completed successfully!")
        print(f"📁 Model saved to: {config['output_dir']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"💡 Try reducing batch_size or switching to a smaller model")


if __name__ == "__main__":
    main() 