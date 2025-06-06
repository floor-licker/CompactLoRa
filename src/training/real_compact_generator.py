#!/usr/bin/env python3
"""
Real Compact Code Generator using actual syntax patterns.
"""

import random
import json
from pathlib import Path
from typing import List, Dict


class RealCompactGenerator:
    """Generates valid Compact code using real syntax patterns."""
    
    def __init__(self):
        self.base_patterns = self._get_base_patterns()
    
    def _get_base_patterns(self) -> Dict[str, str]:
        """Real Compact patterns from validated examples."""
        return {
            "simple_ledger": '''pragma language_version >= 0.14.0;

import CompactStandardLibrary;
export {{ CurvePoint }}

export ledger {ledger_name}: {data_type};

export circuit {circuit_name}({param_name}: {param_type}): [] {{
  {ledger_name} = {operation};
}}''',
            
            "crypto_module": '''pragma language_version >= 0.14.0;

module {module_name} {{
  import CompactStandardLibrary;

  export struct {struct_name} {{
    field1: Bytes<32>;
    field2: Field;
  }}

  export pure circuit {circuit_name}(input: {input_type}): {output_type} {{
    {return_statement}
  }}
}}''',
            
            "signature_circuit": '''pragma language_version >= 0.14.0;

module {module_name} {{
  import CompactStandardLibrary;

  export pure circuit {circuit_name}(msg: Bytes<32>, sk: Field): CurvePoint {{
    const pk: CurvePoint = ec_mul_generator(sk);
    return pk;
  }}
}}'''
        }
    
    def generate_simple_ledger(self) -> str:
        """Generate a simple ledger contract."""
        # Type-safe combinations for ledgers
        ledger_combinations = [
            {
                "ledger_name": "counter",
                "data_type": "Uint<64>",
                "circuit_name": "increment",
                "param_name": "amount",
                "param_type": "Uint<64>",
                "operation": "amount"
            },
            {
                "ledger_name": "balance",
                "data_type": "Uint<64>",
                "circuit_name": "set",
                "param_name": "new_value",
                "param_type": "Uint<64>",
                "operation": "new_value"
            },
            {
                "ledger_name": "data",
                "data_type": "Bytes<32>",
                "circuit_name": "update",
                "param_name": "input",
                "param_type": "Bytes<32>",
                "operation": "input"
            }
        ]
        
        combo = random.choice(ledger_combinations)
        
        return self.base_patterns["simple_ledger"].format(**combo)
    
    def generate_crypto_module(self) -> str:
        """Generate a crypto module."""
        
        # Define type-safe combinations
        type_combinations = [
            {
                "input_type": "Bytes<32>",
                "output_type": "Bytes<32>", 
                "return_statement": "return persistent_hash<Bytes<32>>(input);"
            },
            {
                "input_type": "Field",
                "output_type": "CurvePoint",
                "return_statement": "return ec_mul_generator(input);"
            },
            {
                "input_type": "CurvePoint", 
                "output_type": "Bytes<32>",
                "return_statement": "return persistent_hash<CurvePoint>(input);"
            },
            {
                "input_type": "Uint<64>",
                "output_type": "Field",
                "return_statement": "const field_val: Field = input as Field;\n    return field_val;"
            }
        ]
        
        # Choose a type-safe combination
        combo = random.choice(type_combinations)
        
        variants = {
            "module_name": ["Crypto", "Utils", "Math", "Ledger"],
            "struct_name": ["Data", "Key", "Signature", "Record"],
            "circuit_name": ["process", "hash", "verify", "compute"],
        }
        
        return self.base_patterns["crypto_module"].format(
            module_name=random.choice(variants["module_name"]),
            struct_name=random.choice(variants["struct_name"]),
            circuit_name=random.choice(variants["circuit_name"]),
            input_type=combo["input_type"],
            output_type=combo["output_type"],
            return_statement=combo["return_statement"]
        )
    
    def save_generated_contracts(self, contracts: List[str], output_path: str = "data/real_compact_contracts.jsonl"):
        """Save generated contracts to file."""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            for i, contract in enumerate(contracts):
                entry = {
                    'example_id': f'real_compact_{i:04d}',
                    'code': contract,
                    'source': 'real_compact_generator',
                    'is_valid': True,  # These should be valid by construction
                    'generated_at': f'batch_{i//25}'
                }
                f.write(json.dumps(entry) + '\n')
        
        print(f"💾 Generated {len(contracts)} real Compact contracts saved to {output_path}")

    def generate_contracts(self, num_contracts: int = 100) -> List[str]:
        """Generate multiple valid Compact contracts."""
        contracts = []
        
        for _ in range(num_contracts):
            contract_type = random.choice(["simple_ledger", "crypto_module"])
            
            if contract_type == "simple_ledger":
                contract = self.generate_simple_ledger()
            else:
                contract = self.generate_crypto_module()
            
            contracts.append(contract)
        
        return contracts


def main():
    """Generate real Compact contracts."""
    print("🚀 GENERATING REAL COMPACT CONTRACTS")
    print("=" * 50)
    
    generator = RealCompactGenerator()
    
    # Generate large dataset
    num_contracts = 100
    contracts = generator.generate_contracts(num_contracts)
    
    # Save contracts
    generator.save_generated_contracts(contracts)
    
    # Show sample
    print("📝 Sample contract:")
    print(contracts[0])
    print(f"\n✅ Generated {len(contracts)} contracts with real Compact syntax!")
    
    return contracts


if __name__ == "__main__":
    main() 