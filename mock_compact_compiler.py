#!/usr/bin/env python3
"""
Mock Compact Compiler

Simulates the Compact compiler for testing RL training when the real compiler isn't available.
"""

import random
import re


class MockCompactCompiler:
    """Mock compiler that simulates compilation results."""
    
    def __init__(self, success_rate: float = 0.7):
        self.success_rate = success_rate
    
    def compile_contract(self, contract_code: str) -> bool:
        """Mock compilation that checks basic syntax patterns."""
        
        # Basic syntax checks that a real compiler would catch
        required_patterns = [
            r'pragma language_version',
            r'import CompactStandardLibrary',
            r'export (ledger|circuit|\{)'
        ]
        
        # Check for required patterns
        for pattern in required_patterns:
            if not re.search(pattern, contract_code):
                return False  # Definite compilation failure
        
        # Check for obvious syntax errors
        syntax_errors = [
            r'export ledger\s*;',  # Empty ledger declaration
            r'export circuit\s*\(',  # Incomplete circuit without closing
            r'if\s*\(\s*\)',  # Empty if condition
        ]
        
        for error_pattern in syntax_errors:
            if re.search(error_pattern, contract_code):
                return False
        
        # If code has good complexity indicators, higher chance of success
        complexity_indicators = [
            r'if\s*\(',
            r'else',
            r'const\s+\w+:',
            r'export ledger \w+:',
            r'export circuit \w+\('
        ]
        
        complexity_score = sum(1 for pattern in complexity_indicators 
                             if re.search(pattern, contract_code))
        
        # Higher complexity = higher success rate
        adjusted_success_rate = min(0.95, self.success_rate + (complexity_score * 0.1))
        
        # Random success based on adjusted rate
        return random.random() < adjusted_success_rate


def main():
    """Test the mock compiler."""
    compiler = MockCompactCompiler()
    
    # Test contracts
    good_contract = '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export { CurvePoint }

export ledger state: Field;
export ledger amount: Uint<64>;

export circuit process(new_state: Field, value: Uint<64>): [] {
  if (state == 0) {
    if (value > 100) {
      state = 1;
      amount = value;
    } else {
      state = 2;
    }
  }
}'''

    bad_contract = '''pragma language_version >= 0.14.0;
export ledger ;
export circuit (
'''
    
    print("🧪 MOCK COMPILER TEST")
    print("=" * 30)
    
    print(f"Good Contract Compiles: {compiler.compile_contract(good_contract)}")
    print(f"Bad Contract Compiles: {compiler.compile_contract(bad_contract)}")
    
    # Test multiple times to see randomness
    print(f"\nGood Contract Success Rate (10 trials):")
    successes = sum(1 for _ in range(10) if compiler.compile_contract(good_contract))
    print(f"  {successes}/10 = {successes*10}%")


if __name__ == "__main__":
    main() 