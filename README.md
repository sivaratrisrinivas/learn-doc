# Learn-Doc

> A **Test-Time Training (TTT)** demonstration: the model *learns* from documents during inference, then answers questions **without the document in context**.

---

## 🚀 What This Does

1. **Document Processing** - Parse PDFs/text and chunk into token-aligned segments
2. **TTT Learning** - Model updates weights using masked token prediction (inner-loop)
3. **Context Clearing** - KV cache wiped (proof: no cheating, only learned weights remain)
4. **Interactive Q&A** - Model answers from learned weights only
5. **Session Persistence** - Save/load learned states across sessions

---

## 🎯 Why This Matters

Current LLMs are frozen after training. To use document knowledge:
- **Context stuffing** (expensive, limited to ~128K tokens)
- **Fine-tuning** (slow, needs infrastructure, overwrites knowledge)

**TTT solves this**: the model learns from documents *during inference* without permanent weight changes.

---

## 🛠️ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI

```bash
# Learn from a document
python cli/cli.py learn docs/sample.txt

# Interactive Q&A session
python cli/cli.py interactive

# Full pipeline (learn + interact)
python cli/cli.py run docs/sample.txt

# Reset session for new document
python cli/cli.py reset
```

### 3. Use Your Own Documents

```bash
# Learn from any text or PDF file
python cli/cli.py learn path/to/your/document.pdf
python cli/cli.py learn path/to/your/document.txt
```

---

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `learn <file>` | Learn from a PDF or text file |
| `interactive` | Start Q&A session (after learning) |
| `run <file>` | Full pipeline: learn + interactive |
| `reset` | Clear session for new document |
| `help` | Show help message |

---

## 🏗️ Architecture

```
Document → Extract → Chunk → TTT Learning (Masked Token) → Clear Context → Q&A
                                           ↓
                             W_h ← W_h - η∇L_masked(x, W_h)
```

### Core Components

- **TTT-Linear Layers**: Replace MLPs in transformer blocks 16-23, update gate weights during inference
- **Masked Token Prediction**: Self-supervised learning task forcing document understanding
- **Anchor Regularization**: Balances task loss with regularization to prevent overfitting
- **Base Model**: Qwen2.5-0.5B-Instruct (lightweight, efficient)

### Technical Specifications

| Component | Specification |
|-----------|---------------|
| **Model** | Qwen2.5-0.5B-Instruct |
| **TTT Layers** | Blocks 16-23 (8 layers) |
| **Trainable Weights** | Only `W_h` gate weights in TTT-Linear |
| **Chunk Size** | 512 tokens (training), 2048 tokens (processing) |
| **Learning Rate** | 5e-4 (inner loop) |
| **Epochs** | 20 per document |
| **Regularization** | Anchor regularization (0.01 strength) |

---

## 💡 Recent Improvements & Bug Fixes

### 🔧 Critical Fixes
- **NaN Loss Prevention**: Fixed by casting TTT weights to Float32 (was Float16)
- **Anchor Regularization Balance**: Reduced strength from 1.0 to 0.01 for stable training
- **Gradient Clipping**: Added 0.5 max norm to prevent exploding gradients
- **Memory Management**: CUDA cache clearing between operations

### 📈 Performance Optimizations
- **Simplified Architecture**: Flat structure, removed unused components
- **Session Persistence**: Efficient save/load of TTT states
- **Progress Tracking**: Real-time loss monitoring and progress bars
- **Enhanced CLI**: Better error handling and user feedback

---

## 🧪 Test Documents

### Sample Document (`docs/sample.txt`)
- Simple factual information (AI, France capital, WWW inventor, etc.)
- Tests basic knowledge acquisition

### Test Document (`docs/test_small.txt`)
- Creative fictional content (Crystalline Whales of Jupiter)
- Tests ability to learn novel concepts
- Contains specific facts: Dr. Helena Frost, 2145, gamma-ray communication

---

## 📦 Tech Stack

| Component | Choice | Version |
|-----------|--------|---------|
| **Framework** | PyTorch | ≥2.1.0 |
| **Model** | Qwen2.5-0.5B-Instruct | transformers ≥4.36.0 |
| **PDF Processing** | PyMuPDF | ≥1.23.0 |
| **Tokenization** | tiktoken | latest |
| **Interface** | CLI (Command Line) | - |
| **Acceleration** | Accelerate | latest |

---

## 🔬 How It Works

### 1. Document Processing
- PDF/text parsing with PyMuPDF
- Token-aligned chunking (2048 tokens/chunk)
- Preserves document structure and context

### 2. TTT Learning
- **Inner Loop**: 20 epochs of masked token prediction
- **Weight Updates**: Only gate weights (`W_h`) in TTT-Linear layers
- **Regularization**: Anchor regularization prevents overfitting
- **Stability**: Float32 precision, gradient clipping, NaN detection

### 3. Knowledge Demonstration
- **Context Clearing**: KV cache wiped after learning
- **Weight Persistence**: Learned knowledge stored in TTT weights
- **Interactive Q&A**: Model answers using learned weights only
- **Session Management**: Save/load states across sessions

---

## 📊 Performance Characteristics

- **Memory Usage**: Efficient with CUDA optimization
- **Training Speed**: Fast inner-loop learning (seconds per document)
- **Knowledge Retention**: Proven through context clearing tests
- **Generation Quality**: Coherent answers using learned weights only

---

## 🎯 Key Innovation

This system demonstrates that **Test-Time Training** is practically implementable:

1. **Context Independence**: KV cache cleared, only learned weights remain
2. **Weight Updates**: Measurable changes in TTT layer weights
3. **Knowledge Retention**: Accurate answers without document access
4. **Reversibility**: No permanent model changes, session-based learning

Represents a significant step toward adaptive AI systems that can learn continuously without expensive fine-tuning or context stuffing.

---

## 📚 References

- Sun et al. 2024 - "Learning to (Learn at Test Time)"
- Sara Hooker 2020 - ["The Hardware Lottery"](https://arxiv.org/abs/2009.06489)

