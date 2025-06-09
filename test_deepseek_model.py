#!/usr/bin/env python3
"""Test DeepSeek Coder model for Compact contract generation."""

from rl_lora_complexity_trainer import RLLoRATrainer
import torch

def test_deepseek():
    print('🚀 Testing DeepSeek Coder for Compact contract generation...')
    
    # Initialize trainer with DeepSeek
    trainer = RLLoRATrainer(
        base_model_name="deepseek-ai/deepseek-coder-1.3b-base",
        complexity_threshold=3
    )
    
    # Test generation with proper Compact prompts
    prompts = [
        "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\nexport circuit ComplexLogic {\n",
        "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\nexport ledger StatefulContract {\n",
        "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\nexport module AdvancedLogic {\n"
    ]
    
    print('\n📜 Generated contracts:')
    print('=' * 80)
    
    for i, prompt in enumerate(prompts, 1):
        print(f'\n🔸 Contract {i}:')
        generated = trainer.generate_contract(prompt, max_length=300)
        
        # Analyze complexity
        complexity = trainer.complexity_analyzer.calculate_cyclomatic_complexity(generated)
        compiles = trainer.compiler.compile_contract(generated)
        
        print(f'📋 Generated Code:')
        print('-' * 50)
        print(generated)
        print('-' * 50)
        print(f'🧮 Complexity: {complexity}')
        print(f'✅ Compiles: {compiles}')
        print()
    
    # Run a quick evaluation
    print('\n📊 Running evaluation...')
    results = trainer.evaluate_model(num_samples=5)
    
    return results

if __name__ == '__main__':
    test_deepseek() 