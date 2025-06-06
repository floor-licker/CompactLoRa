# CompactLoRa

**LoRA Fine-Tuning for Compact Smart Contract Generation**

A comprehensive system for training AI models and generating valid **Compact smart contracts** using both template-based generation and LoRA (Low-Rank Adaptation) fine-tuning approaches.

## 🎯 Project Goals

### Primary Objective
Generate syntactically correct and semantically valid **Compact smart contracts** that:
- Compile successfully with the official Compact compiler
- Exhibit realistic business logic patterns
- Have configurable **cyclomatic complexity** for thorough testing
- Scale from simple educational examples to production-ready contracts

### Secondary Objectives
1. **Create High-Quality Training Data**: Build a validated dataset of Compact contracts for ML training
2. **Hybrid Generation Approach**: Combine template-based generation (immediate) with LoRA fine-tuning (scalable)
3. **Quality Assurance**: Ensure 100% compilation success rate through real compiler validation
4. **Complexity Control**: Generate contracts with specific cyclomatic complexity requirements

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Template-Based │    │   ML-Enhanced   │    │   Validation    │
│   Generation    │───▶│   Generation    │───▶│    Pipeline     │
│                 │    │   (LoRA)        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Pattern-Based  │    │  Fine-tuned     │    │  Compiler       │
│  Contracts      │    │  AI Models      │    │  Verification   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tools & Technologies

### **1. Compact Compiler (`compactc`)**
**Purpose**: Official Midnight Protocol compiler for validation
- **Location**: `/Users/juliustranquilli/webisoft/compactc_v0.22.0_x86_64-apple-darwin/compactc`
- **Usage**: Real-time validation of generated contracts
- **Integration**: Automated testing pipeline ensures 100% compilation success

### **2. LoRA (Low-Rank Adaptation) Fine-Tuning**
**Purpose**: Efficient fine-tuning of large language models
- **Library**: HuggingFace PEFT (Parameter Efficient Fine-Tuning)
- **Base Models**: GPT-2, DialoGPT for code generation
- **Efficiency**: Only 0.65% of parameters modified, 65-82% loss reduction achieved
- **Target Modules**: `["c_attn", "c_proj"]` for transformer attention layers

### **3. Dataset Management**
**Purpose**: High-quality training data curation
- **HuggingFace Datasets**: For efficient data loading and processing
- **Validation Pipeline**: Real compiler integration for quality assurance
- **Current Size**: 340 validated training examples from 118 unique contracts

### **4. Python Ecosystem**
- **PyTorch**: Deep learning framework for model training
- **Transformers**: Pre-trained model loading and tokenization
- **Datasets**: Data management and preprocessing
- **argparse**: CLI interface for flexible usage

## 🧠 Technical Implementation

### **Template-Based Generation System**

#### **Pattern Architecture**
```python
class CompactContractGenerator:
    def __init__(self):
        self.ledger_patterns = [...]  # Ledger-based contracts
        self.crypto_patterns = [...]  # Cryptographic modules
```

#### **Complexity-Aware Templates**
Each template is designed with specific cyclomatic complexity:

```python
def get_pattern_complexity(self, pattern: dict) -> int:
    """Calculate cyclomatic complexity = 1 + number of decision points"""
    template = pattern["template"]
    if_count = template.count(" if ")  # Count decision points
    return 1 + if_count
```

### **Cyclomatic Complexity Generation**

#### **Complexity Levels**

| Complexity | Pattern Type | Decision Points | Example |
|------------|-------------|-----------------|---------|
| **1** | Simple | 0 | `counter = amount;` |
| **2** | Validated | 1 | Hash validation with single `if` |
| **3** | Complex | 2 | Nested validation with bounds checking |

#### **Low-Level Implementation**

**Simple Contract (Complexity = 1)**:
```compact
export circuit increment(amount: Uint<64>): [] {
  counter = amount;  // Linear execution, no branches
}
```

**Complex Contract (Complexity = 3)**:
```compact
export circuit safe_increment(amount: Uint<64>): [] {
  if (amount > 0) {                    // Decision point 1
    if (counter + amount <= max_value) { // Decision point 2  
      counter = counter + amount;       // Path A
    } else {
      counter = max_value;              // Path B
    }
  }                                     // Path C (implicit else)
}
// Total paths: 3, Complexity = 1 + 2 decisions = 3
```

#### **Type-Safe Pattern Generation**
```python
# Ensure type compatibility to prevent compilation errors
ledger_combinations = [
    {
        "ledger_name": "counter",
        "data_type": "Uint<64>",        # Type-safe
        "param_type": "Uint<64>",       # Compatible parameter
        "operation": "amount"           # Valid assignment
    }
]
```

### **Validation Pipeline**

#### **Real-Time Compiler Integration**
```python
def validate_contract(self, code: str) -> bool:
    """Validate using actual Compact compiler"""
    with tempfile.NamedTemporaryFile(suffix='.compact') as f:
        f.write(code)
        result = subprocess.run([
            '/path/to/compactc', f.name, temp_dir
        ], capture_output=True)
        return result.returncode == 0  # Success = exit code 0
```

#### **Quality Metrics**
- **Template Success Rate**: 100% (57/100 initially → 100/100 after fixes)
- **Compilation Success**: 100% validation through real compiler
- **Dataset Quality**: 340 examples, average 347 characters per contract

### **LoRA Fine-Tuning Pipeline**

#### **Model Configuration**
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # Low rank for efficiency
    lora_alpha=16,           # Scaling parameter  
    lora_dropout=0.1,        # Regularization
    target_modules=["c_attn", "c_proj"],  # Attention layers
    bias="none"
)
```

#### **Training Results**
| Model | Dataset Size | Loss Reduction | Trainable Parameters |
|-------|-------------|----------------|---------------------|
| GPT-2 | 340 examples | 65-82% | 0.65% of total params |

### **Complexity Filtering System**

#### **CLI Integration**
```bash
# Generate contracts with minimum complexity
python3 generate_compact_contracts.py --min-complexity 2

# View patterns by complexity
python3 generate_compact_contracts.py --list-complex
```

#### **Pattern Selection Algorithm**
```python
def generate_complex_contract(self, min_complexity: int) -> str:
    # Filter patterns by complexity requirement
    patterns = [p for p in all_patterns 
               if self.get_pattern_complexity(p) >= min_complexity]
    
    if not patterns:
        raise ValueError(f"No patterns with complexity >= {min_complexity}")
    
    return random.choice(patterns)["template"]
```

## 📊 Results & Achievements

### **Generation Success Metrics**
- **Template Accuracy**: 100% compilation success rate
- **Complexity Range**: Supports complexity levels 1-3+
- **Pattern Diversity**: 6 ledger + 6 crypto patterns = 12 total variations
- **Real-World Validation**: All contracts tested with official Compact compiler

### **Dataset Statistics**
```
📈 Training Dataset:
├── 340 total training examples
├── 118 unique validated contracts
├── Sources:
│   ├── 13 original valid contracts
│   ├── 5 legacy generated contracts  
│   └── 100 new template-generated contracts
└── Average contract length: 347 characters
```

### **Training Performance**
- **Loss Reduction**: 65-82% across different model configurations
- **Parameter Efficiency**: Only 811,008 trainable parameters vs 125M total
- **Training Speed**: ~80 seconds for 10 epochs on CPU

## 🚀 Usage Examples

### **Basic Contract Generation**
```bash
# Generate a simple contract
python3 generate_compact_contracts.py

# Generate with specific pattern
python3 generate_compact_contracts.py --type ledger --pattern counter

# Generate complex contracts only
python3 generate_compact_contracts.py --min-complexity 2 --count 5
```

### **Complexity Analysis**
```bash
# View all patterns with complexity info
python3 generate_compact_contracts.py --list-complex

# Output:
# Ledger Patterns:
#   • counter (complexity: 1): Simple counter ledger
#   • validated_counter (complexity: 3): Counter with validation logic
#   • conditional_balance (complexity: 3): Balance with conditional updates
```

### **Batch Generation**
```bash
# Generate multiple contracts and save to file
python3 generate_compact_contracts.py \
  --min-complexity 2 \
  --count 10 \
  --output complex_contracts.compact
```

## 🔬 Advanced Features

### **Hybrid Approach Benefits**
1. **Immediate Deployment**: Template system provides instant contract generation
2. **Scalable Learning**: LoRA fine-tuning enables learning from larger datasets
3. **Quality Assurance**: Real compiler validation ensures production readiness
4. **Complexity Control**: Precise control over code complexity for testing needs

### **Future Enhancements**
- **Pattern Expansion**: Add more contract types (DeFi, NFT, governance)
- **Advanced Complexity**: Support higher complexity levels (4-10+)
- **Semantic Validation**: Beyond syntax, validate business logic correctness
- **Multi-Language Support**: Extend to other blockchain languages

## 📁 Project Structure

```
CompactLoRa/                         # 56 files total (vs ~200+ before)
├── README.md                        # Comprehensive documentation
├── requirements.txt                 # Dependencies
├── generate_compact_contracts.py    # 🎯 MAIN PRODUCTION TOOL
├── src/training/                    # Core training modules (4 files)
│   ├── lora_trainer.py             # LoRA fine-tuning
│   ├── real_compact_generator.py   # Template engine
│   ├── validate_and_combine.py     # Dataset pipeline
│   └── direct_compact_test.py      # Testing utilities
├── data/                           # Clean datasets
│   ├── compact_final_dataset/      # 340 validated examples
│   ├── real_compact_contracts.jsonl # Generated contracts
│   └── training_corpus.jsonl      # Original dataset
└── models/                         # Final trained model only
    └── compact-lora-gpt2-final/    # Best performing model
```

## 🎯 Key Innovations

1. **Real Compiler Integration**: First system to validate generated contracts with actual Compact compiler
2. **Complexity-Aware Generation**: Explicit control over cyclomatic complexity for systematic testing
3. **Type-Safe Templates**: Guaranteed compilation through careful type system design
4. **Hybrid ML Approach**: Combines deterministic templates with learned patterns
5. **Production-Ready Output**: 100% compilation success rate for immediate deployment

---

**This system represents a complete pipeline from raw contract patterns to production-ready Compact smart contracts, with rigorous validation and complexity control for comprehensive blockchain development needs.**

## ⚡ Quick Start

### **Generate Your First Contract**
```bash
# Clone the repository
git clone https://github.com/floor-licker/CompactLoRa.git
cd CompactLoRa

# Generate a simple contract
python3 generate_compact_contracts.py

# Generate complex contracts with validation logic
python3 generate_compact_contracts.py --min-complexity 2 --count 3

# Save contracts to file
python3 generate_compact_contracts.py --type crypto --output my_contracts.compact
```

### **Explore Available Patterns**
```bash
# List all patterns with complexity info
python3 generate_compact_contracts.py --list-complex

# Generate only high-complexity contracts
python3 generate_compact_contracts.py --min-complexity 3
```

### **Validate Generated Contracts**
```bash
# Contracts are automatically validated, but you can test manually:
/path/to/compactc generated_contract.compact output_dir/
```

## 📞 Support & Documentation

- **Compact Language**: [Midnight Protocol Documentation](https://docs.midnight.network/)
- **LoRA Training**: [HuggingFace PEFT](https://huggingface.co/docs/peft/)
- **Cyclomatic Complexity**: [Code Complexity Theory](https://en.wikipedia.org/wiki/Cyclomatic_complexity)

---

**🎯 Ready to generate production-quality Compact smart contracts with guaranteed compilation success!** 