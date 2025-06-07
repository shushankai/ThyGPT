# Transformer Language Model for Character-Level Text Generation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

This repository contains a PyTorch implementation of a transformer-based language model for character-level text generation. The model can be trained on any text corpus to generate new text in the same style.

## Features

- Character-level text generation
- Transformer architecture with multi-head attention
- Positional embeddings for sequence order
- Residual connections and layer normalization
- Dropout regularization
- Efficient training on GPU
- Text generation with adjustable context window

## Model Architecture

The model consists of the following components:

1. **Token Embeddings**: Converts character indices to dense vectors
2. **Position Embeddings**: Encodes positional information
3. **Transformer Blocks** (6 layers):
   - Multi-head self-attention (6 heads)
   - Position-wise feed-forward network
   - Residual connections
   - Layer normalization
4. **Output Projection**: Converts final embeddings to vocabulary logits

```
Input
  │
▼
Token Embedding
  │
▼
Position Embedding (added)
  │
▼
[Transformer Block] × 6
  │
▼
Layer Normalization
  │
▼
Output Projection
  │
▼
Logits
```

## Requirements

- Python 3.7+
- PyTorch 1.12+
- CUDA (for GPU acceleration)

## Installation

```bash
pip install torch
```

## Usage

### 1. Prepare Dataset

Place your text file in the project directory named `input.txt`. The model will train on this text.

### 2. Training the Model

Run the script to train the model:

```bash
python transformer_lm.py
```

The training process will display progress every 200 steps:

```
Loading dataset...
Initializing model...
Model parameters: 3.65M
Starting training...
Step 0: Train loss 4.4123, Val loss 4.4112
Step 200: Train loss 2.1256, Val loss 2.1325
...
Step 4800: Train loss 1.1023, Val loss 1.2015
```

### 3. Generating Text

After training, the model will generate 1000 characters of text:

```
Generating text...
TROIUS:
I am not so well: I thank you: good my lord,
How does your honour for this many a day?
I have, my lord, I thank your lordship, I have
The heart to do a thing that may be done
To the great charge of the people and the state,
And therefore I will not be so far to say
The king is not so much as I am sure
He is a man of such a simple truth
That he will not be so far to say
The king is not so much as I am sure
He is a man of such a simple truth
That he will not be so far to say
The king is not so much as I am sure
He is a man of such a simple truth
...
```

### 4. Custom Generation

To generate text with a custom starting prompt:

```python
context = torch.tensor([encode("KING:")], dtype=torch.long, device=DEVICE)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
```

## Configuration

You can modify these hyperparameters in the script:

```python
# Hyperparameters
BATCH_SIZE = 64            # Sequences per batch
BLOCK_SIZE = 256           # Context window length
N_EMB = 384                # Embedding dimension
N_HEADS = 6                # Attention heads
DROPOUT = 0.2              # Dropout probability
LEARNING_RATE = 3e-4       # Learning rate
TRAIN_ITERATIONS = 5000    # Training steps
N_LAYERS = 6               # Transformer blocks
```

## Key Components

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Combines multiple attention heads and projects output.
    
    Args:
        num_heads (int): Number of parallel attention heads
        head_size (int): Dimensionality of each attention head
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMB, N_EMB)
        self.dropout = nn.Dropout(DROPOUT)
```

### Transformer Block

```python
class Block(nn.Module):
    """
    Transformer block with attention, feedforward, and residual connections.
    
    Args:
        n_emb (int): Embedding dimension
        n_head (int): Number of attention heads
    """
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_emb)
```

## Performance Notes

- Training on CPU: ~20 minutes for 5000 iterations
- Training on GPU (NVIDIA RTX 3080): ~3 minutes for 5000 iterations
- Model size: ~3.65 million parameters

## Future Improvements

1. Implement learning rate scheduling
2. Add gradient clipping
3. Implement beam search for generation
4. Add model checkpointing
5. Implement sub-word tokenization
6. Add support for larger datasets

