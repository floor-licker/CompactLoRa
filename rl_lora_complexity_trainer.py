#!/usr/bin/env python3
"""
RL + LoRA Complexity Trainer

Reinforcement Learning system using LoRA to train models to generate
Compact contracts with:
- Cyclomatic complexity > 3
- 100% compilation success
"""

import torch
import subprocess
import tempfile
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random


@dataclass
class RewardMetrics:
    """Reward calculation for generated contracts."""
    cyclomatic_complexity: int
    compiles: bool
    reward_score: float


class CompactCompiler:
    """Interface to Compact compiler for validation."""
    
    def __init__(self, compiler_path: str = "compactc"):
        self.compiler_path = compiler_path
        self.use_mock = False
        
        # Check if real compiler is available
        try:
            result = subprocess.run([compiler_path, "--version"], 
                                  capture_output=True, timeout=5)
            if result.returncode != 0:
                self.use_mock = True
                print("⚠️  Compact compiler version check failed, using mock compiler")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.use_mock = True
            print("⚠️  Compact compiler not found, using mock compiler")
            
        if self.use_mock:
            from mock_compact_compiler import MockCompactCompiler
            self.mock_compiler = MockCompactCompiler(success_rate=0.6)
        else:
            print(f"✅ Using real Compact compiler: {compiler_path}")
    
    def compile_contract(self, contract_code: str) -> bool:
        """Test if contract compiles successfully."""
        if self.use_mock:
            return self.mock_compiler.compile_contract(contract_code)
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.compact', delete=False) as f:
                f.write(contract_code)
                temp_file = f.name
            
            # Create output directory for compiled artifacts
            output_dir = tempfile.mkdtemp()
            
            # Run compactc compiler with correct syntax
            # compactc <input.compact> <output_directory>
            result = subprocess.run(
                [self.compiler_path, temp_file, output_dir],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Compilation error: {e}")
            return False


class ComplexityAnalyzer:
    """Analyze cyclomatic complexity of generated contracts."""
    
    def calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate McCabe cyclomatic complexity."""
        # Count decision points in Compact syntax
        if_count = code.count(' if (')
        else_count = code.count(' else ')
        
        # Base complexity is 1, each decision point adds 1
        complexity = 1 + if_count + else_count
        return complexity
    
    def meets_complexity_threshold(self, code: str, threshold: int = 3) -> bool:
        """Check if code meets minimum complexity threshold."""
        return self.calculate_cyclomatic_complexity(code) > threshold


class RLLoRATrainer:
    """Reinforcement Learning trainer using LoRA for complexity optimization."""
    
    def __init__(self, 
                 base_model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 complexity_threshold: int = 3):
        
        self.base_model_name = base_model_name
        self.complexity_threshold = complexity_threshold
        
        # Load tokenizer and model
        print(f"🔄 Loading DeepSeek Coder model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Configure LoRA for DeepSeek Coder
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # DeepSeek Coder specific
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, self.lora_config)
        
        # Initialize tools
        self.compiler = CompactCompiler()
        self.complexity_analyzer = ComplexityAnalyzer()
        
        print(f"✅ Initialized RL+LoRA trainer with DeepSeek Coder")
        print(f"🎯 Complexity threshold: > {complexity_threshold}")
        print(f"🔧 LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    
    def calculate_reward(self, generated_code: str) -> RewardMetrics:
        """Calculate reward based on complexity and compilation success."""
        
        # Check compilation
        compiles = self.compiler.compile_contract(generated_code)
        
        # Calculate complexity
        complexity = self.complexity_analyzer.calculate_cyclomatic_complexity(generated_code)
        
        # Reward function:
        # - Base reward: 1.0 if compiles, 0.0 if not
        # - Complexity bonus: exponential reward for complexity > threshold
        if compiles:
            base_reward = 1.0
            if complexity > self.complexity_threshold:
                # Exponential bonus for higher complexity
                complexity_bonus = (complexity - self.complexity_threshold) ** 1.5
                total_reward = base_reward + complexity_bonus
            else:
                # Penalty for low complexity
                total_reward = 0.5
        else:
            # No reward for non-compiling code
            total_reward = 0.0
        
        return RewardMetrics(
            cyclomatic_complexity=complexity,
            compiles=compiles,
            reward_score=total_reward
        )
    
    def generate_contract(self, prompt: str, max_length: int = 400) -> str:
        """Generate a contract using current model."""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)
        
        # Generation parameters optimized for code
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,  # Slightly lower for more structured code
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition in code
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def collect_experience(self, num_samples: int = 32) -> List[Dict]:
        """Collect experience samples for RL training."""
        
        # Enhanced prompts with better Compact structure
        base_prompts = [
            "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\n// Generate a Compact smart contract with multiple conditional branches\nexport circuit ComplexContract {\n",
            "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\n// Create a Compact ledger with complex logic and multiple if-else statements\nexport ledger StatefulContract {\n",
            "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\n// Implement a Compact contract with loops and conditional logic\nexport module AdvancedLogic {\n"
        ]
        
        experiences = []
        
        for i in range(num_samples):
            prompt = random.choice(base_prompts)
            
            # Generate contract
            generated_code = self.generate_contract(prompt)
            
            # Calculate reward
            reward_metrics = self.calculate_reward(generated_code)
            
            experiences.append({
                'prompt': prompt,
                'generated_code': generated_code,
                'reward': reward_metrics.reward_score,
                'complexity': reward_metrics.cyclomatic_complexity,
                'compiles': reward_metrics.compiles
            })
            
            print(f"Sample {i+1}/{num_samples}: Reward={reward_metrics.reward_score:.2f}, "
                  f"Complexity={reward_metrics.cyclomatic_complexity}, "
                  f"Compiles={reward_metrics.compiles}")
        
        return experiences
    
    def train_with_rl(self, 
                     num_iterations: int = 10,
                     samples_per_iteration: int = 32,
                     save_path: str = "models/rl-lora-compact"):
        """Train model using reinforcement learning with LoRA."""
        
        print(f"🚀 Starting RL training for {num_iterations} iterations")
        print(f"🎯 Target: Complexity > {self.complexity_threshold} + Compilation Success")
        
        best_avg_reward = 0.0
        
        for iteration in range(num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}")
            
            # Collect experience
            experiences = self.collect_experience(samples_per_iteration)
            
            # Filter for positive rewards (compiling, complex contracts)
            positive_experiences = [exp for exp in experiences if exp['reward'] > 1.0]
            
            if positive_experiences:
                # Create training data from positive experiences
                training_texts = [exp['generated_code'] for exp in positive_experiences]
                
                # Fine-tune LoRA weights on successful examples
                self._fine_tune_on_examples(training_texts)
                
                avg_reward = sum(exp['reward'] for exp in experiences) / len(experiences)
                success_rate = sum(1 for exp in experiences if exp['compiles']) / len(experiences)
                avg_complexity = sum(exp['complexity'] for exp in experiences) / len(experiences)
                
                print(f"📊 Iteration {iteration + 1} Results:")
                print(f"   • Average Reward: {avg_reward:.3f}")
                print(f"   • Compilation Success Rate: {success_rate:.1%}")
                print(f"   • Average Complexity: {avg_complexity:.1f}")
                print(f"   • Positive Examples: {len(positive_experiences)}")
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_model(f"{save_path}-best")
                    print(f"💾 Saved new best model (reward: {avg_reward:.3f})")
            
            else:
                print("❌ No positive examples found in this iteration")
        
        # Save final model
        self.save_model(save_path)
        print(f"\n✅ Training completed! Final model saved to {save_path}")
        print(f"🏆 Best average reward achieved: {best_avg_reward:.3f}")
    
    def _fine_tune_on_examples(self, training_texts: List[str]):
        """Fine-tune LoRA weights on positive examples."""
        
        # Tokenize training examples
        tokenized_inputs = []
        for text in training_texts:
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            tokenized_inputs.append(tokens)
        
        # Simple gradient update (in practice, you'd use a proper RL algorithm like PPO)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for tokens in tokenized_inputs:
            outputs = self.model(**tokens, labels=tokens['input_ids'])
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.model.eval()
    
    def save_model(self, save_path: str):
        """Save the LoRA model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def evaluate_model(self, num_samples: int = 20) -> Dict:
        """Evaluate current model performance."""
        
        print(f"🧪 Evaluating model on {num_samples} samples...")
        
        experiences = self.collect_experience(num_samples)
        
        # Calculate metrics
        total_reward = sum(exp['reward'] for exp in experiences)
        compilation_success = sum(1 for exp in experiences if exp['compiles'])
        complex_contracts = sum(1 for exp in experiences if exp['complexity'] > self.complexity_threshold)
        high_quality = sum(1 for exp in experiences if exp['compiles'] and exp['complexity'] > self.complexity_threshold)
        
        results = {
            'avg_reward': total_reward / num_samples,
            'compilation_rate': compilation_success / num_samples,
            'complexity_rate': complex_contracts / num_samples,
            'high_quality_rate': high_quality / num_samples,
            'avg_complexity': sum(exp['complexity'] for exp in experiences) / num_samples
        }
        
        print(f"📊 Evaluation Results:")
        print(f"   • Average Reward: {results['avg_reward']:.3f}")
        print(f"   • Compilation Rate: {results['compilation_rate']:.1%}")
        print(f"   • Complexity > {self.complexity_threshold} Rate: {results['complexity_rate']:.1%}")
        print(f"   • High Quality Rate: {results['high_quality_rate']:.1%}")
        print(f"   • Average Complexity: {results['avg_complexity']:.1f}")
        
        return results


def main():
    """Main training loop for RL + LoRA complexity trainer."""
    
    print("🤖 RL + LORA COMPLEXITY TRAINER")
    print("=" * 60)
    print("🎯 Goal: Generate contracts with complexity > 3 that compile")
    print("⚡ Method: Reinforcement Learning + LoRA fine-tuning")
    
    # Initialize trainer
    trainer = RLLoRATrainer(complexity_threshold=3)
    
    # Evaluate baseline
    print("\n📊 BASELINE EVALUATION")
    baseline_results = trainer.evaluate_model(num_samples=10)
    
    # Train with RL
    print("\n🚀 STARTING RL TRAINING")
    trainer.train_with_rl(
        num_iterations=5,
        samples_per_iteration=16,
        save_path="models/rl-lora-complexity"
    )
    
    # Final evaluation
    print("\n📊 FINAL EVALUATION")
    final_results = trainer.evaluate_model(num_samples=20)
    
    # Show improvement
    print("\n📈 IMPROVEMENT SUMMARY")
    print(f"Compilation Rate: {baseline_results['compilation_rate']:.1%} → {final_results['compilation_rate']:.1%}")
    print(f"Complexity Rate: {baseline_results['complexity_rate']:.1%} → {final_results['complexity_rate']:.1%}")
    print(f"High Quality Rate: {baseline_results['high_quality_rate']:.1%} → {final_results['high_quality_rate']:.1%}")


if __name__ == "__main__":
    main() 