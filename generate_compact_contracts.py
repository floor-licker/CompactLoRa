#!/usr/bin/env python3
"""
Production Compact Contract Generator

Generates valid Compact smart contracts using proven templates.
"""

import random
import argparse
from typing import List


class CompactContractGenerator:
    """Production-ready Compact contract generator."""
    
    def __init__(self):
        self.ledger_patterns = [
            {
                "name": "counter",
                "description": "Simple counter ledger",
                "template": '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export { CurvePoint }

export ledger counter: Uint<64>;

export circuit increment(amount: Uint<64>): [] {
  counter = amount;
}'''
            },
            {
                "name": "validated_counter",
                "description": "Counter with validation logic",
                "template": '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export { CurvePoint }

export ledger counter: Uint<64>;
export ledger max_value: Uint<64>;

export circuit safe_increment(amount: Uint<64>): [] {
  if (amount > 0) {
    if (counter + amount <= max_value) {
      counter = counter + amount;
    } else {
      counter = max_value;
    }
  }
}'''
            },
            {
                "name": "balance", 
                "description": "Balance tracking ledger",
                "template": '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export { CurvePoint }

export ledger balance: Uint<64>;

export circuit set(new_value: Uint<64>): [] {
  balance = new_value;
}'''
            },
            {
                "name": "conditional_balance",
                "description": "Balance with conditional updates",
                "template": '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export { CurvePoint }

export ledger balance: Uint<64>;
export ledger min_balance: Uint<64>;

export circuit update_balance(new_value: Uint<64>, allow_negative: Field): [] {
  if (allow_negative == 1) {
    balance = new_value;
  } else {
    if (new_value >= min_balance) {
      balance = new_value;
    } else {
      balance = min_balance;
    }
  }
}'''
            },
            {
                "name": "data_store",
                "description": "Data storage ledger", 
                "template": '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export { CurvePoint }

export ledger data: Bytes<32>;

export circuit update(input: Bytes<32>): [] {
  data = input;
}'''
            },
            {
                "name": "validated_storage",
                "description": "Data storage with validation",
                "template": '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export { CurvePoint }

export ledger data: Bytes<32>;
export ledger data_hash: Bytes<32>;
export ledger is_valid: Field;

export circuit store_with_validation(input: Bytes<32>, expected_hash: Bytes<32>): [] {
  const computed_hash: Bytes<32> = persistent_hash<Bytes<32>>(input);
  if (computed_hash == expected_hash) {
    data = input;
    data_hash = computed_hash;
    is_valid = 1;
  } else {
    is_valid = 0;
  }
}'''
            }
        ]
        
        self.crypto_patterns = [
            {
                "name": "hash_module",
                "description": "Cryptographic hashing module",
                "template": '''pragma language_version >= 0.14.0;

module Crypto {
  import CompactStandardLibrary;

  export pure circuit hash_bytes(input: Bytes<32>): Bytes<32> {
    return persistent_hash<Bytes<32>>(input);
  }
}'''
            },
            {
                "name": "conditional_hash",
                "description": "Hash with input validation",
                "template": '''pragma language_version >= 0.14.0;

module Crypto {
  import CompactStandardLibrary;

  export pure circuit secure_hash(input: Bytes<32>, salt: Field): Bytes<32> {
    if (salt > 0) {
      const salted_input: Bytes<32> = persistent_hash<Field>(salt);
      const combined: Bytes<32> = persistent_hash<Bytes<32>>(input);
      return persistent_hash<Bytes<32>>(combined);
    } else {
      return persistent_hash<Bytes<32>>(input);
    }
  }
}'''
            },
            {
                "name": "key_generation",
                "description": "Public key generation module",
                "template": '''pragma language_version >= 0.14.0;

module KeyGen {
  import CompactStandardLibrary;

  export pure circuit generate_pubkey(private_key: Field): CurvePoint {
    return ec_mul_generator(private_key);
  }
}'''
            },
            {
                "name": "validated_keygen",
                "description": "Key generation with validation",
                "template": '''pragma language_version >= 0.14.0;

module KeyGen {
  import CompactStandardLibrary;

  export pure circuit safe_generate_pubkey(private_key: Field, min_key: Field): CurvePoint {
    if (private_key >= min_key) {
      if (private_key > 0) {
        return ec_mul_generator(private_key);
      } else {
        return ec_mul_generator(min_key);
      }
    } else {
      return ec_mul_generator(min_key);
    }
  }
}'''
            },
            {
                "name": "field_conversion",
                "description": "Type conversion module",
                "template": '''pragma language_version >= 0.14.0;

module Utils {
  import CompactStandardLibrary;

  export pure circuit uint_to_field(input: Uint<64>): Field {
    const field_val: Field = input as Field;
    return field_val;
  }
}'''
            },
            {
                "name": "conditional_conversion",
                "description": "Type conversion with bounds checking",
                "template": '''pragma language_version >= 0.14.0;

module Utils {
  import CompactStandardLibrary;

  export pure circuit safe_uint_to_field(input: Uint<64>, max_val: Uint<64>): Field {
    if (input <= max_val) {
      if (input > 0) {
        const field_val: Field = input as Field;
        return field_val;
      } else {
        const default_val: Field = 1 as Field;
        return default_val;
      }
    } else {
      const capped_val: Field = max_val as Field;
      return capped_val;
    }
  }
}'''
            }
        ]
    
    def generate_ledger_contract(self, pattern_name: str = None) -> str:
        """Generate a ledger-based contract."""
        if pattern_name:
            pattern = next((p for p in self.ledger_patterns if p["name"] == pattern_name), None)
            if not pattern:
                raise ValueError(f"Unknown ledger pattern: {pattern_name}")
        else:
            pattern = random.choice(self.ledger_patterns)
        
        return pattern["template"]
    
    def generate_crypto_module(self, pattern_name: str = None) -> str:
        """Generate a crypto module contract."""
        if pattern_name:
            pattern = next((p for p in self.crypto_patterns if p["name"] == pattern_name), None)
            if not pattern:
                raise ValueError(f"Unknown crypto pattern: {pattern_name}")
        else:
            pattern = random.choice(self.crypto_patterns)
        
        return pattern["template"]
    
    def generate_random_contract(self) -> str:
        """Generate a random valid contract."""
        contract_type = random.choice(["ledger", "crypto"])
        
        if contract_type == "ledger":
            return self.generate_ledger_contract()
        else:
            return self.generate_crypto_module()
    
    def get_pattern_complexity(self, pattern: dict) -> int:
        """Calculate cyclomatic complexity of a pattern."""
        template = pattern["template"]
        # Count decision points: if statements
        if_count = template.count(" if ")
        # Each if adds +1 to complexity
        return 1 + if_count
    
    def generate_complex_contract(self, min_complexity: int = 2, contract_type: str = "random") -> str:
        """Generate a contract with minimum cyclomatic complexity."""
        if contract_type == "ledger":
            patterns = [p for p in self.ledger_patterns if self.get_pattern_complexity(p) >= min_complexity]
        elif contract_type == "crypto":
            patterns = [p for p in self.crypto_patterns if self.get_pattern_complexity(p) >= min_complexity]
        else:  # random
            all_patterns = self.ledger_patterns + self.crypto_patterns
            patterns = [p for p in all_patterns if self.get_pattern_complexity(p) >= min_complexity]
        
        if not patterns:
            raise ValueError(f"No patterns found with complexity >= {min_complexity}")
        
        pattern = random.choice(patterns)
        return pattern["template"]
    
    def list_patterns_with_complexity(self) -> None:
        """List all available patterns with their complexity."""
        print("📋 Available Ledger Patterns:")
        for pattern in self.ledger_patterns:
            complexity = self.get_pattern_complexity(pattern)
            print(f"   • {pattern['name']} (complexity: {complexity}): {pattern['description']}")
        
        print("\n📋 Available Crypto Patterns:")
        for pattern in self.crypto_patterns:
            complexity = self.get_pattern_complexity(pattern)
            print(f"   • {pattern['name']} (complexity: {complexity}): {pattern['description']}")
    
    def list_patterns(self) -> None:
        """List all available patterns."""
        print("📋 Available Ledger Patterns:")
        for pattern in self.ledger_patterns:
            print(f"   • {pattern['name']}: {pattern['description']}")
        
        print("\n📋 Available Crypto Patterns:")
        for pattern in self.crypto_patterns:
            print(f"   • {pattern['name']}: {pattern['description']}")
    
    def generate_multiple(self, count: int, contract_type: str = "random") -> List[str]:
        """Generate multiple contracts."""
        contracts = []
        
        for _ in range(count):
            if contract_type == "ledger":
                contract = self.generate_ledger_contract()
            elif contract_type == "crypto":
                contract = self.generate_crypto_module()
            else:  # random
                contract = self.generate_random_contract()
            
            contracts.append(contract)
        
        return contracts

    def generate_complex_contract_with_bias(self, min_complexity: int = 2, contract_type: str = "random", complexity_bias: float = 1.0) -> str:
        """Generate a contract with minimum cyclomatic complexity and bias toward higher complexity."""
        if contract_type == "ledger":
            patterns = [p for p in self.ledger_patterns if self.get_pattern_complexity(p) >= min_complexity]
        elif contract_type == "crypto":
            patterns = [p for p in self.crypto_patterns if self.get_pattern_complexity(p) >= min_complexity]
        else:  # random
            all_patterns = self.ledger_patterns + self.crypto_patterns
            patterns = [p for p in all_patterns if self.get_pattern_complexity(p) >= min_complexity]
        
        if not patterns:
            raise ValueError(f"No patterns found with complexity >= {min_complexity}")
        
        if complexity_bias <= 1.0:
            # No bias, use original random selection
            pattern = random.choice(patterns)
        else:
            # Apply complexity bias
            weights = []
            for pattern in patterns:
                complexity = self.get_pattern_complexity(pattern)
                weight = complexity ** complexity_bias  # Exponential bias
                weights.append(weight)
            
            # Weighted random selection
            pattern = random.choices(patterns, weights=weights)[0]
        
        return pattern["template"]


def main():
    """CLI interface for contract generation."""
    parser = argparse.ArgumentParser(description='Generate Compact smart contracts')
    parser.add_argument('--type', choices=['ledger', 'crypto', 'random'], default='random',
                       help='Type of contract to generate')
    parser.add_argument('--pattern', type=str, help='Specific pattern name to use')
    parser.add_argument('--count', type=int, default=1, help='Number of contracts to generate')
    parser.add_argument('--list', action='store_true', help='List available patterns')
    parser.add_argument('--list-complex', action='store_true', help='List patterns with complexity info')
    parser.add_argument('--min-complexity', type=int, default=1, help='Minimum cyclomatic complexity (default: 1)')
    parser.add_argument('--complexity-bias', type=float, default=1.0, help='Bias toward complex patterns (1.0=no bias, 2.0=quadratic bias)')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    generator = CompactContractGenerator()
    
    if args.list:
        generator.list_patterns()
        return
    
    if args.list_complex:
        generator.list_patterns_with_complexity()
        return
    
    print("🚀 COMPACT CONTRACT GENERATOR")
    print("=" * 40)
    print(f"🎯 Minimum Complexity: {args.min_complexity}")
    if args.complexity_bias > 1.0:
        print(f"📊 Complexity Bias: {args.complexity_bias}x")
    
    if args.count == 1:
        # Generate single contract
        if args.pattern:
            # Specific pattern requested
            if args.type == "ledger":
                contract = generator.generate_ledger_contract(args.pattern)
            elif args.type == "crypto":
                contract = generator.generate_crypto_module(args.pattern)
            else:
                contract = generator.generate_random_contract()
        else:
            # Generate with complexity filter and bias
            if args.min_complexity > 1 or args.complexity_bias > 1.0:
                contract = generator.generate_complex_contract_with_bias(args.min_complexity, args.type, args.complexity_bias)
            else:
                if args.type == "ledger":
                    contract = generator.generate_ledger_contract()
                elif args.type == "crypto":
                    contract = generator.generate_crypto_module()
                else:
                    contract = generator.generate_random_contract()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(contract)
            print(f"✅ Contract saved to {args.output}")
        else:
            print("📝 Generated Contract:")
            print(contract)
    
    else:
        # Generate multiple contracts
        contracts = []
        for _ in range(args.count):
            if args.min_complexity > 1 or args.complexity_bias > 1.0:
                contract = generator.generate_complex_contract_with_bias(args.min_complexity, args.type, args.complexity_bias)
            else:
                contracts_batch = generator.generate_multiple(1, args.type)
                contract = contracts_batch[0]
            contracts.append(contract)
        
        if args.output:
            with open(args.output, 'w') as f:
                for i, contract in enumerate(contracts):
                    f.write(f"// Contract {i+1}\n")
                    f.write(contract)
                    f.write("\n\n" + "="*50 + "\n\n")
            print(f"✅ {len(contracts)} contracts saved to {args.output}")
        else:
            for i, contract in enumerate(contracts):
                print(f"\n📝 Contract {i+1}:")
                print(contract)
                if i < len(contracts) - 1:
                    print("\n" + "-"*40)
    
    print(f"\n🎯 Generated {args.count} valid Compact contract{'s' if args.count > 1 else ''}!")
    if args.min_complexity > 1:
        print(f"📊 All contracts have cyclomatic complexity >= {args.min_complexity}")
    if args.complexity_bias > 1.0:
        print(f"⚡ Applied {args.complexity_bias}x bias toward complex patterns")


if __name__ == "__main__":
    main() 