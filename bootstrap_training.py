#!/usr/bin/env python3
"""Bootstrap training strategy for CompactLoRa."""

from rl_lora_complexity_trainer import RLLoRATrainer
import json
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

class BootstrapTrainer:
    """Progressive training starting with simple goals."""
    
    def __init__(self):
        print('🚀 Initializing Bootstrap Training Strategy')
        
        # Load our real validated contracts for fine-tuning
        self.load_real_contracts()
        
        # Initialize trainer with lower complexity threshold
        self.trainer = RLLoRATrainer(
            base_model_name="deepseek-ai/deepseek-coder-1.3b-base",
            complexity_threshold=1  # Start with complexity > 1
        )
    
    def load_real_contracts(self):
        """Load our validated contracts."""
        print('📚 Loading real validated contracts...')
        
        with open('data/real_compact_contracts.jsonl', 'r') as f:
            self.real_contracts = [json.loads(line) for line in f.readlines()]
        
        print(f'✅ Loaded {len(self.real_contracts)} validated contracts')
    
    def phase_1_syntax_training(self, iterations=3):
        """Phase 1: Learn basic Compact syntax (goal: just compile)."""
        print('\n🎯 PHASE 1: Basic Syntax Training')
        print('Goal: Generate contracts that compile (any complexity)')
        
        # Temporarily lower the complexity threshold to 0 (any compilable contract is good)
        self.trainer.complexity_threshold = 0
        
        # Train with small samples to get compilation success
        self.trainer.train_with_rl(
            num_iterations=iterations,
            samples_per_iteration=8,  # Smaller batches for faster iteration
            save_path="models/deepseek-phase1-syntax"
        )
    
    def phase_2_complexity_training(self, iterations=5):
        """Phase 2: Increase complexity while maintaining compilation."""
        print('\n🎯 PHASE 2: Complexity Training')
        print('Goal: Generate contracts with complexity > 2 that compile')
        
        # Load the Phase 1 model
        print('🔄 Loading Phase 1 model...')
        self.trainer.model = AutoPeftModelForCausalLM.from_pretrained("models/deepseek-phase1-syntax")
        self.trainer.tokenizer = AutoTokenizer.from_pretrained("models/deepseek-phase1-syntax")
        
        if self.trainer.tokenizer.pad_token is None:
            self.trainer.tokenizer.pad_token = self.trainer.tokenizer.eos_token
        
        # Increase complexity requirement
        self.trainer.complexity_threshold = 2
        
        # Train for higher complexity
        self.trainer.train_with_rl(
            num_iterations=iterations,
            samples_per_iteration=12,
            save_path="models/deepseek-phase2-complexity"
        )
    
    def phase_3_advanced_training(self, iterations=5):
        """Phase 3: Target high complexity (> 3)."""
        print('\n🎯 PHASE 3: Advanced Complexity Training')
        print('Goal: Generate contracts with complexity > 3 that compile')
        
        # Load the Phase 2 model
        print('🔄 Loading Phase 2 model...')
        self.trainer.model = AutoPeftModelForCausalLM.from_pretrained("models/deepseek-phase2-complexity")
        self.trainer.tokenizer = AutoTokenizer.from_pretrained("models/deepseek-phase2-complexity")
        
        if self.trainer.tokenizer.pad_token is None:
            self.trainer.tokenizer.pad_token = self.trainer.tokenizer.eos_token
        
        # Final complexity target
        self.trainer.complexity_threshold = 3
        
        # Train for final complexity goal
        self.trainer.train_with_rl(
            num_iterations=iterations,
            samples_per_iteration=16,
            save_path="models/deepseek-final-complex"
        )
    
    def run_bootstrap_training(self):
        """Run the complete 3-phase bootstrap training."""
        print('🎯 BOOTSTRAP TRAINING STRATEGY')
        print('=' * 60)
        print('Phase 1: Learn Compact syntax (compilation focus)')
        print('Phase 2: Add complexity while compiling')  
        print('Phase 3: Achieve high complexity + compilation')
        print()
        
        # Run all phases
        self.phase_1_syntax_training(iterations=3)
        self.phase_2_complexity_training(iterations=5)
        self.phase_3_advanced_training(iterations=5)
        
        print('\n🎉 BOOTSTRAP TRAINING COMPLETE!')
        print('Final model saved to: models/deepseek-final-complex')
        
        # Final evaluation
        print('\n📊 FINAL EVALUATION')
        final_trainer = RLLoRATrainer(complexity_threshold=3)
        final_trainer.model = AutoPeftModelForCausalLM.from_pretrained("models/deepseek-final-complex")
        final_trainer.tokenizer = AutoTokenizer.from_pretrained("models/deepseek-final-complex")
        
        if final_trainer.tokenizer.pad_token is None:
            final_trainer.tokenizer.pad_token = final_trainer.tokenizer.eos_token
        
        results = final_trainer.evaluate_model(num_samples=20)
        
        return results

def main():
    bootstrap = BootstrapTrainer()
    results = bootstrap.run_bootstrap_training()
    
    print('\n📈 FINAL RESULTS SUMMARY')
    print(f'Compilation Rate: {results["compilation_rate"]:.1%}')
    print(f'High Complexity Rate: {results["complexity_rate"]:.1%}')
    print(f'High Quality Rate: {results["high_quality_rate"]:.1%}')

if __name__ == '__main__':
    main() 