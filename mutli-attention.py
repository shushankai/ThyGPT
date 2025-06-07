import torch
from torch import nn
import torch.nn.functional as F

# ------------------------- Configuration -------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available for faster computation
SEED = 1337  # Fixed random seed for reproducibility
torch.manual_seed(SEED)  # Set PyTorch random seed

# Hyperparameters
BATCH_SIZE = 64            # Number of sequences processed in parallel
BLOCK_SIZE = 256           # Context window length (max tokens considered for prediction)
N_EMB = 384                # Embedding dimension (size of token representations)
N_HEADS = 6                # Number of attention heads in multi-head attention
DROPOUT = 0.2              # Dropout probability for regularization
LEARNING_RATE = 3e-4       # Learning rate for Adam optimizer
TRAIN_ITERATIONS = 5000    # Total training steps
EVAL_INTERVAL = 200        # Evaluate loss every N steps
N_LAYERS = 6               # Number of Transformer blocks in the model

# ------------------------- Dataset Preparation -------------------------
print("Loading dataset...")
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()  # Read entire text file into memory

# Create character-level vocabulary
vocab = sorted(set(text))  # Unique characters in text
vocab_size = len(vocab)
# Create mapping dictionaries
stoi = {ch: i for i, ch in enumerate(vocab)}  # string to index (character → token ID)
itos = {i: ch for ch, i in stoi.items()}      # index to string (token ID → character)
# Encoder/decoder functions
encode = lambda s: [stoi[c] for c in s]       # string → list of token IDs
decode = lambda l: ''.join([itos[i] for i in l])  # list of token IDs → string

# Tokenize dataset and split into train/validation sets
data = torch.tensor(encode(text), dtype=torch.long)  # Convert entire text to tensor of token IDs
train_data = data[:int(0.9 * len(data))]  # First 90% for training
val_data = data[int(0.9 * len(data)):]    # Last 10% for validation

# ------------------------- Batch Loader -------------------------
def get_batch(split):
    """
    Generate a batch of input and target sequences.
    
    Args:
        split (str): 'train' or 'val' to select dataset
    
    Returns:
        xb (Tensor): Input token sequences [BATCH_SIZE, BLOCK_SIZE]
        yb (Tensor): Target token sequences [BATCH_SIZE, BLOCK_SIZE]
    """
    source = train_data if split == 'train' else val_data
    # Random starting indices for sequences
    ix = torch.randint(0, len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    # Input sequences (from ix to ix+BLOCK_SIZE)
    xb = torch.stack([source[i:i+BLOCK_SIZE] for i in ix])
    # Target sequences (shifted by one character)
    yb = torch.stack([source[i+1:i+BLOCK_SIZE+1] for i in ix])
    return xb.to(DEVICE), yb.to(DEVICE)

# ------------------------- Loss Estimation -------------------------
@torch.no_grad()  # Disable gradient calculation for evaluation
def estimate_loss():
    """
    Estimate average loss for train and validation datasets.
    
    Returns:
        dict: Average losses for 'train' and 'val' splits
    """
    model.eval()  # Set model to evaluation mode
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_INTERVAL)
        for i in range(EVAL_INTERVAL):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()  # Return to training mode
    return out

# ------------------------- Attention Head -------------------------
class Head(nn.Module):
    """
    Single causal self-attention head with masking to prevent lookahead.
    
    Args:
        head_size (int): Dimensionality of this attention head
    """
    def __init__(self, head_size):
        super().__init__()
        # Linear projections for key, query, value
        self.key = nn.Linear(N_EMB, head_size, bias=False)    # [N_EMB → head_size]
        self.query = nn.Linear(N_EMB, head_size, bias=False)  # [N_EMB → head_size]
        self.value = nn.Linear(N_EMB, head_size, bias=False)  # [N_EMB → head_size]
        self.dropout = nn.Dropout(DROPOUT)
        # Causal mask (lower triangular matrix) to prevent attention to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        """
        Forward pass for attention head.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_length, N_EMB]
            
        Returns:
            Tensor: Output tensor [batch_size, seq_length, head_size]
        """
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dim
        
        # Project inputs to key, query, value
        k = self.key(x)     # [B, T, head_size]
        q = self.query(x)   # [B, T, head_size]
        v = self.value(x)   # [B, T, head_size]
        
        # Compute attention scores (dot product scaled by sqrt(d_k))
        # [B, T, head_size] @ [B, head_size, T] → [B, T, T]
        att = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        
        # Apply causal mask to prevent attention to future tokens
        # Mask shape: [T, T] → broadcast to [B, T, T]
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Softmax to get attention weights
        att = F.softmax(att, dim=-1)  # [B, T, T]
        att = self.dropout(att)  # Regularization
        
        # Weighted sum of values
        return att @ v  # [B, T, head_size]

# ------------------------- Multi-Head Attention -------------------------
class MultiHeadAttention(nn.Module):
    """
    Combines multiple attention heads and projects output.
    
    Args:
        num_heads (int): Number of parallel attention heads
        head_size (int): Dimensionality of each attention head
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create multiple attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Linear projection to combine head outputs
        self.proj = nn.Linear(N_EMB, N_EMB)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Forward pass for multi-head attention.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_length, N_EMB]
            
        Returns:
            Tensor: Output tensor [batch_size, seq_length, N_EMB]
        """
        # Concatenate outputs from all heads
        # Each head: [B, T, head_size] → concat → [B, T, num_heads * head_size]
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # Project back to embedding dimension and apply dropout
        return self.dropout(self.proj(out))  # [B, T, N_EMB]

# ------------------------- Feed Forward Network -------------------------
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with expansion and projection.
    
    Args:
        n_emb (int): Embedding dimension (input and output size)
    """
    def __init__(self, n_emb):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),  # Expand dimension
            nn.ReLU(),                    # Non-linearity
            nn.Linear(4 * n_emb, n_emb),  # Project back to original dimension
            nn.Dropout(DROPOUT)           # Regularization
        )

    def forward(self, x):
        """
        Forward pass for feed-forward network.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_length, n_emb]
            
        Returns:
            Tensor: Output tensor [batch_size, seq_length, n_emb]
        """
        return self.ff(x)

# ------------------------- Transformer Block -------------------------
class Block(nn.Module):
    """
    Transformer block with attention, feedforward, residual connections, and layer normalization.
    
    Args:
        n_emb (int): Embedding dimension
        n_head (int): Number of attention heads
    """
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head  # Size per attention head
        # Layer normalizations
        self.ln1 = nn.LayerNorm(n_emb)  # Pre-attention normalization
        self.ln2 = nn.LayerNorm(n_emb)  # Pre-FFN normalization
        # Attention and feed-forward components
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_emb)

    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_length, n_emb]
            
        Returns:
            Tensor: Output tensor [batch_size, seq_length, n_emb]
        """
        # Self-attention with residual connection
        # x shape: [B, T, N_EMB]
        x = x + self.sa(self.ln1(x))  # Residual + LayerNorm + Attention
        # Feed-forward with residual connection
        x = x + self.ff(self.ln2(x))  # Residual + LayerNorm + Feedforward
        return x

# ------------------------- Language Model -------------------------
class BigramLanguageModel(nn.Module):
    """
    Transformer-based language model for character-level text generation.
    Architecture:
    1. Token embeddings
    2. Position embeddings
    3. Stacked transformer blocks
    4. Final layer normalization
    5. Language modeling head
    """
    def __init__(self):
        super().__init__()
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, N_EMB)  # [vocab_size → N_EMB]
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMB)  # [position index → N_EMB]
        
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(N_EMB, N_HEADS) for _ in range(N_LAYERS)]  # Stack N_LAYERS blocks
        )
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(N_EMB)  # Final layer normalization
        self.lm_head = nn.Linear(N_EMB, vocab_size)  # Project to vocabulary size

    def forward(self, idx, targets=None):
        """
        Forward pass through language model.
        
        Args:
            idx (Tensor): Input token indices [batch_size, seq_length]
            targets (Tensor): Target token indices [batch_size, seq_length]
            
        Returns:
            logits (Tensor): Prediction logits [batch_size, seq_length, vocab_size]
            loss (Tensor): Cross-entropy loss (if targets provided)
        """
        B, T = idx.shape  # Batch size, Sequence length
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)  # [B, T] → [B, T, N_EMB]
        
        # Position embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))  # [T] → [T, N_EMB] → [B, T, N_EMB] via broadcast
        x = tok_emb + pos_emb  # Combine embeddings
        
        # Process through transformer blocks
        x = self.blocks(x)  # [B, T, N_EMB]
        
        # Final layer norm and projection
        x = self.ln_f(x)  # [B, T, N_EMB]
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten sequence dimension
            logits = logits.view(B * T, vocab_size)  # [B*T, vocab_size]
            targets = targets.view(-1)  # [B*T]
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, context, max_new_tokens):
        """
        Generate new tokens given a context sequence.
        
        Args:
            context (Tensor): Initial token sequence [1, seq_length]
            max_new_tokens (int): Number of new tokens to generate
            
        Returns:
            Tensor: Generated sequence [1, seq_length + max_new_tokens]
        """
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Start with initial context
            for _ in range(max_new_tokens):
                # Crop context to last BLOCK_SIZE tokens (model's context window)
                context_crop = context[:, -BLOCK_SIZE:]  # [1, min(seq_length, BLOCK_SIZE)]
                
                # Get model predictions
                logits, _ = self(context_crop)  # [1, T, vocab_size]
                
                # Focus on last token's prediction
                probs = F.softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
                
                # Sample next token from probability distribution
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                
                # Append to sequence
                context = torch.cat((context, next_token), dim=1)  # [1, seq_length+1]
        
        return context

# ------------------------- Training Setup -------------------------
print("Initializing model...")
model = BigramLanguageModel().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# AdamW optimizer (improves generalization over standard Adam)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ------------------------- Training Loop -------------------------
print("Starting training...")
for step in range(TRAIN_ITERATIONS):
    # Periodically evaluate on train/validation sets
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

    # Get a batch of data
    xb, yb = get_batch('train')  # xb: [BATCH_SIZE, BLOCK_SIZE], yb: [BATCH_SIZE, BLOCK_SIZE]
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)  # Clear gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

# ------------------------- Text Generation -------------------------
print("\nGenerating text...")
# Start with a single start token (0 = first character in vocabulary)
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

# Generate 1000 new tokens
generated = model.generate(context, max_new_tokens=1000)

# Decode token IDs to characters
print(decode(generated[0].tolist()))