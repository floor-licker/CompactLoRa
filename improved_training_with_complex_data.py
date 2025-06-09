#!/usr/bin/env python3
"""Improved training using complex real-world Compact contracts."""

import json
from rl_lora_complexity_trainer import RLLoRATrainer, ComplexityAnalyzer, CompactCompiler

def load_complex_training_data():
    """Load the rich training corpus with battleship, crypto, etc."""
    print('📚 Loading complex training corpus...')
    
    contracts = []
    complexity_analyzer = ComplexityAnalyzer()
    
    with open('data/training_corpus.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('is_valid', False):  # Only use valid contracts
                code = data['code']
                complexity = complexity_analyzer.calculate_cyclomatic_complexity(code)
                
                contracts.append({
                    'code': code,
                    'complexity': complexity,
                    'source': data.get('source_filename', 'unknown'),
                    'compiles': data.get('compilation_result', {}).get('success', False)
                })
    
    print(f'✅ Loaded {len(contracts)} complex contracts')
    
    # Show complexity distribution
    by_complexity = {}
    by_source = {}
    
    for contract in contracts:
        complexity = contract['complexity']
        source = contract['source']
        
        by_complexity[complexity] = by_complexity.get(complexity, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
    
    print('\n📊 Complexity Distribution:')
    for complexity in sorted(by_complexity.keys()):
        print(f'  Complexity {complexity}: {by_complexity[complexity]} contracts')
    
    print('\n📊 Source Distribution:')
    for source, count in sorted(by_source.items()):
        print(f'  {source}: {count} contracts')
    
    return contracts

def train_with_complex_data():
    """Train the model using complex real-world data."""
    print('🚀 Starting improved training with complex contracts...')
    
    # Load complex data
    training_data = load_complex_training_data()
    
    # Filter for good examples (use both compiling and non-compiling with good syntax)
    high_complexity = [c for c in training_data if c['complexity'] >= 3]
    compiling_examples = [c for c in training_data if c['compiles']]
    
    # Use complex examples even if they don't compile (good for learning syntax)
    good_examples = high_complexity + compiling_examples
    # Remove duplicates
    seen_codes = set()
    good_examples = [c for c in good_examples if c['code'] not in seen_codes and not seen_codes.add(c['code'])]
    
    print(f'🎯 Using {len(good_examples)} contracts ({len(high_complexity)} complex + {len(compiling_examples)} compiling)')
    
    # Show examples
    print('\n🔍 Sample complex contracts:')
    for i, contract in enumerate(good_examples[:3]):
        print(f'\n--- Example {i+1}: {contract["source"]} (complexity {contract["complexity"]}) ---')
        print(contract['code'][:200] + '...' if len(contract['code']) > 200 else contract['code'])
    
    # Initialize trainer
    trainer = RLLoRATrainer(
        base_model_name="deepseek-ai/deepseek-coder-1.3b-base",
        complexity_threshold=2  # Lower threshold since we have good examples
    )
    
    # Fine-tune on complex examples first (supervised learning)
    print('\n🎓 Fine-tuning on complex contracts...')
    complex_codes = [c['code'] for c in good_examples]
    trainer.fine_tune_on_examples(complex_codes[:min(10, len(complex_codes))])  # Use best examples
    
    # Then do RL training
    print('\n🎮 Starting RL training with complexity rewards...')
    trainer.train_with_rl(
        num_iterations=10,
        samples_per_iteration=32,
        save_path="models/deepseek-complex-rl"
    )
    
    print('✅ Training complete! Model saved to models/deepseek-complex-rl')

if __name__ == "__main__":
    train_with_complex_data() 