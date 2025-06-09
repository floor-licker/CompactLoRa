#!/usr/bin/env python3
"""Test real validated contracts."""

import json
from rl_lora_complexity_trainer import CompactCompiler, ComplexityAnalyzer

def test_real_contracts():
    print('🧪 Testing real validated contracts...')
    
    compiler = CompactCompiler()
    complexity_analyzer = ComplexityAnalyzer()
    
    # Load first few real contracts
    with open('data/real_compact_contracts.jsonl', 'r') as f:
        contracts = [json.loads(line) for line in f.readlines()[:5]]
    
    for i, contract in enumerate(contracts):
        print(f'\n🔸 Contract {i+1}:')
        print('-' * 50)
        print(contract['code'])
        print('-' * 50)
        
        complexity = complexity_analyzer.calculate_cyclomatic_complexity(contract['code'])
        compiles = compiler.compile_contract(contract['code'])
        
        print(f'🧮 Complexity: {complexity}')
        print(f'✅ Compiles: {compiles}')
        print(f'📝 Source: {contract["source"]}')
        print(f'✓ Marked as valid: {contract["is_valid"]}')

if __name__ == '__main__':
    test_real_contracts() 