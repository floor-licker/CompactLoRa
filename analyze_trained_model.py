#!/usr/bin/env python3
"""Analyze the trained DeepSeek model output."""

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
from rl_lora_complexity_trainer import ComplexityAnalyzer, CompactCompiler

def analyze_model():
    print('🔍 Analyzing trained DeepSeek model...')
    
    # Load the trained model
    model = AutoPeftModelForCausalLM.from_pretrained('models/deepseek-rl-complexity')
    tokenizer = AutoTokenizer.from_pretrained('models/deepseek-rl-complexity')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize analyzers
    complexity_analyzer = ComplexityAnalyzer()
    compiler = CompactCompiler()
    
    # Test prompts
    prompts = [
        "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\nexport circuit ComplexLogic {\n",
        "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\nexport ledger StatefulContract {\n",
        "pragma language_version >= 0.14.0;\n\nimport CompactStandardLibrary;\n\nexport module AdvancedLogic {\n"
    ]
    
    print('\n📜 Sample generations from trained model:')
    print('=' * 80)
    
    for i, prompt in enumerate(prompts, 1):
        print(f'\n🔸 Sample {i}:')
        
        # Generate
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=400,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Analyze
        complexity = complexity_analyzer.calculate_cyclomatic_complexity(generated)
        compiles = compiler.compile_contract(generated)
        
        print(f'📋 Generated Code:')
        print('-' * 50)
        print(generated)
        print('-' * 50)
        print(f'🧮 Complexity: {complexity}')
        print(f'✅ Compiles: {compiles}')
        
        if not compiles:
            print('❌ Compilation issues likely:')
            print('   - Missing closing braces }')
            print('   - Invalid Compact syntax')
            print('   - Incomplete contract structure')
        
        print()
    
    # Compare with valid examples
    print('\n📚 Compare with valid Compact contract:')
    print('-' * 50)
    valid_example = """pragma language_version >= 0.14.0;

import CompactStandardLibrary;

export ledger BalanceTracker {
    public state: HashMap<Address, UInt256> = HashMap();
    
    endpoint deposit() public {
        let sender = msg.sender;
        let amount = msg.value;
        if (amount > 0) {
            let current = state.get(sender);
            if (current.is_some()) {
                state.set(sender, current.unwrap() + amount);
            } else {
                state.set(sender, amount);
            }
        }
    }
}"""
    
    print(valid_example)
    print('-' * 50)
    valid_complexity = complexity_analyzer.calculate_cyclomatic_complexity(valid_example)
    valid_compiles = compiler.compile_contract(valid_example)
    print(f'🧮 Valid Complexity: {valid_complexity}')
    print(f'✅ Valid Compiles: {valid_compiles}')

if __name__ == '__main__':
    analyze_model() 