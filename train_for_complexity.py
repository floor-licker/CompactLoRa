#!/usr/bin/env python3
"""
Complexity Analysis for RL Training

Focused complexity analysis for RL+LoRA training of Compact contracts.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ComplexityMetrics:
    """Metrics for measuring contract complexity."""
    cyclomatic_complexity: int
    nesting_depth: int
    num_variables: int
    num_circuits: int


class ComplexityAnalyzer:
    """Analyze complexity for RL reward calculations."""
    
    def analyze_contract_complexity(self, contract_code: str) -> ComplexityMetrics:
        """Analyze the complexity of a contract for RL rewards."""
        return ComplexityMetrics(
            cyclomatic_complexity=self._analyze_cyclomatic_complexity(contract_code),
            nesting_depth=self._analyze_nesting_depth(contract_code),
            num_variables=self._count_variables(contract_code),
            num_circuits=self._count_circuits(contract_code)
        )
    
    def _analyze_cyclomatic_complexity(self, code: str) -> int:
        """Calculate McCabe cyclomatic complexity."""
        # Count decision points
        if_count = code.count(' if (')
        else_count = code.count(' else ')
        # Base complexity is 1, each decision point adds 1
        return 1 + if_count + else_count
    
    def _analyze_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0
        
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.endswith('{'):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped == '}':
                current_depth -= 1
        
        return max_depth
    
    def _count_variables(self, code: str) -> int:
        """Count ledger variables and circuit parameters."""
        ledger_count = code.count('export ledger')
        circuit_params = code.count(': Field') + code.count(': Uint<') + code.count(': CurvePoint')
        return ledger_count + circuit_params
    
    def _count_circuits(self, code: str) -> int:
        """Count the number of circuits."""
        return code.count('export circuit')
    
    def meets_complexity_threshold(self, code: str, threshold: int = 3) -> bool:
        """Check if code meets minimum complexity threshold."""
        return self._analyze_cyclomatic_complexity(code) > threshold


def main():
    """Test complexity analysis."""
    analyzer = ComplexityAnalyzer()
    
    # Test with simple contract
    simple_contract = '''pragma language_version >= 0.14.0;
export ledger counter: Uint<64>;
export circuit increment(amount: Uint<64>): [] {
  counter = amount;
}'''
    
    # Test with complex contract  
    complex_contract = '''pragma language_version >= 0.14.0;
export ledger state: Field;
export ledger amount: Uint<64>;
export ledger owner: CurvePoint;
export circuit process(new_state: Field, value: Uint<64>): [] {
  if (state == 0) {
    if (value > 100) {
      if (amount + value <= 1000) {
        state = 1;
        amount = amount + value;
      } else {
        state = 2;
      }
    } else {
      state = 3;
    }
  }
}'''
    
    print("🧪 COMPLEXITY ANALYSIS TEST")
    print("=" * 40)
    
    for i, (name, contract) in enumerate([("Simple", simple_contract), ("Complex", complex_contract)]):
        metrics = analyzer.analyze_contract_complexity(contract)
        meets_threshold = analyzer.meets_complexity_threshold(contract, 3)
        
        print(f"\n📝 {name} Contract:")
        print(f"   • Cyclomatic Complexity: {metrics.cyclomatic_complexity}")
        print(f"   • Nesting Depth: {metrics.nesting_depth}")
        print(f"   • Variables: {metrics.num_variables}")
        print(f"   • Circuits: {metrics.num_circuits}")
        print(f"   • Meets Threshold (>3): {'✅' if meets_threshold else '❌'}")


if __name__ == "__main__":
    main() 