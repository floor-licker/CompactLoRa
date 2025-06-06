#!/usr/bin/env python3
"""
Validate generated contracts and create final high-quality dataset.
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset


class CompactDatasetValidator:
    """Validates and combines Compact contract datasets."""
    
    def __init__(self, compiler_path: str = "/Users/juliustranquilli/webisoft/compactc_v0.22.0_x86_64-apple-darwin/compactc"):
        self.compiler_path = Path(compiler_path)
        
    def validate_contract(self, code: str) -> bool:
        """Validate a single contract using the Compact compiler."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.compact', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Create temp output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run compiler with correct arguments
                result = subprocess.run(
                    [str(self.compiler_path), temp_file, temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                return result.returncode == 0
            
        except Exception:
            return False
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def validate_and_combine_datasets(self) -> Dataset:
        """Validate all contracts and create final dataset."""
        print("🔄 Validating and combining Compact datasets...")
        
        all_valid_contracts = []
        
        # 1. Load original valid contracts
        original_path = Path("data/training_corpus.jsonl")
        if original_path.exists():
            with open(original_path, 'r') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        if example.get('is_valid', False):
                            all_valid_contracts.append(example['code'])
        
        print(f"📚 Loaded {len(all_valid_contracts)} original valid contracts")
        
        # 2. Load and validate generated contracts (old Solidity-style)
        old_generated_path = Path("data/generated_contracts.jsonl")
        if old_generated_path.exists():
            old_generated_valid = 0
            with open(old_generated_path, 'r') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        code = example['code']
                        
                        # Validate each generated contract
                        if self.validate_contract(code):
                            all_valid_contracts.append(code)
                            old_generated_valid += 1
            
            print(f"✅ Validated {old_generated_valid} old generated contracts")
        
        # 3. Load and validate new real Compact contracts
        real_compact_path = Path("data/real_compact_contracts.jsonl")
        if real_compact_path.exists():
            real_compact_valid = 0
            with open(real_compact_path, 'r') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        code = example['code']
                        
                        # Validate each real compact contract
                        if self.validate_contract(code):
                            all_valid_contracts.append(code)
                            real_compact_valid += 1
            
            print(f"✅ Validated {real_compact_valid} real Compact contracts")
        
        # 4. Create diverse training examples
        training_examples = []
        
        for code in all_valid_contracts:
            # Pure code examples (best for learning syntax)
            training_examples.append({
                "text": code.strip()
            })
            
            # Code completion examples
            lines = code.split('\n')
            if len(lines) > 5:
                for split_ratio in [0.3, 0.7]:
                    split_point = int(len(lines) * split_ratio)
                    if 2 <= split_point < len(lines) - 1:
                        prefix = '\n'.join(lines[:split_point])
                        completion = '\n'.join(lines[split_point:])
                        
                        training_examples.append({
                            "text": f"{prefix}\n{completion}"
                        })
        
        print(f"✅ Created {len(training_examples)} high-quality training examples")
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(training_examples)
        return dataset
    
    def save_final_dataset(self, dataset: Dataset, output_path: str = "data/compact_final_dataset"):
        """Save the final validated dataset."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        dataset.save_to_disk(str(output_path))
        
        # Save sample
        sample_path = output_path / "sample_examples.json"
        sample_data = dataset.select(range(min(5, len(dataset))))
        with open(sample_path, 'w') as f:
            json.dump(sample_data.to_list(), f, indent=2)
        
        print(f"💾 Final dataset saved to {output_path}")
        print(f"📄 Sample examples saved to {sample_path}")
        
        # Print statistics
        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Total examples: {len(dataset)}")
        print(f"   Average text length: {sum(len(ex['text']) for ex in dataset) / len(dataset):.0f} chars")
        
        return output_path


def main():
    """Create the final validated dataset."""
    print("🚀 CREATING FINAL VALIDATED COMPACT DATASET")
    print("=" * 55)
    
    validator = CompactDatasetValidator()
    dataset = validator.validate_and_combine_datasets()
    output_path = validator.save_final_dataset(dataset)
    
    print(f"\n✅ Final high-quality dataset ready!")
    print(f"📁 Dataset location: {output_path}")
    print(f"🎯 This dataset contains only validated, compilable Compact code")


if __name__ == "__main__":
    main() 