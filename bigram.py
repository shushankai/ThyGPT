import torch
from pathlib import Path
from torchsummary import summary
from torch import nn
import torch.nn.functional as F

# ------------------------- Hyperparameters -------------------------
SEED = 1333                 # Random seed for reproducibility
BATCH_SIZE = 32             # Number of sequences per batch
BLOCK_SIZE = 8              # Context window length (how many tokens to consider at a time)
TRAIN_ITERATION = 10000     # Number of training steps
EVAL_ITERATION = 200        # Number of evaluation steps
TRAIN_VAL_SPLIT = 0.9       # Ratio to split data into training and validation
LEARNING_RATE  = 1e-3       # Learning rate for optimizer
# Device setup for Efficiency
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------- Dataset Loading -------------------------
print("-------------------------- Getting the Dataset --------------------------")
# Download the dataset from GitHub (TinyShakespeare)
# !curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Read the text data
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# ------------------------- Vocabulary Creation -------------------------
# Get all unique characters in the text to create vocabulary
vocab = sorted(set(text))
vocab_size = len(vocab)

# Create mappings from char to index and index to char
stoi = {ch: idx for idx, ch in enumerate(vocab)}  # String to Integer
itos = {idx: ch for ch, idx in stoi.items()}      # Integer to String

# Helper encoding and decoding functions
encode = lambda s: [stoi[c] for c in s]           # String to list of integers
decode = lambda l: "".join([itos[i] for i in l])  # List of integers to string

# Encode the full text into a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

# Split the data into train and validation sets
split_index = int(TRAIN_VAL_SPLIT * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

print('-------------------------- Data Loaded successfully! --------------------------')

# ------------------------- Batch Generator -------------------------
def get_batch(split: str, batch_size: int, block_size: int, device: torch.device):
    """
    Generates a batch of inputs (x) and targets (y) for training or validation.

    Args:
        split (str): 'train' or 'val' to select dataset.
        batch_size (int): Number of sequences in the batch.
        block_size (int): Length of each sequence.
        device (torch.device): Device on which data should be loaded

    Returns:
        tuple: A tuple of (input tensor, target tensor), each of shape [batch_size, block_size]
    """
    data = train_data if split == "train" else val_data
    max_index = len(data) - block_size - 1
    ix = torch.randint(0, max_index, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

# ------------------------- Estimate Loss ---------------------------
@torch.no_grad()
def estimate_loss():
    """
    Estimate the average loss on both the Training and validation dataset
    
    Returns:
        dict: A dictionary with keys 'train' and 'val' where each value is the 
        average over 'EVAL_ITERATION' batches.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERATION)
        for k in range(EVAL_ITERATION):
            xb, yb = get_batch(split, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
            
        out[split] = losses.mean()
    model.train()
    return out
        
        
    


# ------------------------- Model Definition -------------------------
class BigramLanguageModel(nn.Module):
    """
    A simple Bigram Language Model using token embeddings.
    Predicts the next character given the current one.
    """
    def __init__(self, vocab_size: int):
        """
        Initialize the model with a learnable embedding table.

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
        """
        super().__init__()
        # Each token maps directly to a vector of size vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (B, T) with token indices.
            targets (Tensor, optional): Target tensor of shape (B, T).

        Returns:
            tuple: (logits, loss) where
                logits -> Tensor of shape (B*T, vocab_size)
                loss   -> Cross entropy loss (if targets is not None)
        """
        # Get logits from embedding: shape (B, T, vocab_size)
        logits = self.token_embeddings(x)

        loss = None
        if targets is not None:
            # Reshape logits and targets to match for cross entropy loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, max_length: int, context: torch.Tensor):
        """
        Generate a sequence of tokens from the model.

        Args:
            max_length (int): Maximum number of new tokens to generate.
            context (Tensor): Tensor of shape (1, T) with starting indices.

        Returns:
            Tensor: A tensor of shape (1, T + max_length) with generated token indices.
        """
        for _ in range(max_length):
            # Forward pass to get logits
            logits, _ = self(context)

            # Focus on the last timestep's logits
            logits = logits[:, -1, :]  # shape (B=1, vocab_size)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the context
            context = torch.cat((context, next_token), dim=1)

        return context
    
# ------------------------- Model Initialization ----------------------
print('-------------------------- Model Initialized --------------------------')
model = BigramLanguageModel(vocab_size=vocab_size).to(device=DEVICE)
# print(summary(model, input_size=(BLOCK_SIZE,), batch_size=1, device=str(DEVICE)))

# ------------------------- Optimizer Setup ---------------------------
optimizer = torch.optim.AdamW(params= model.parameters(), lr = LEARNING_RATE)

# ------------------------- Training Loop ----------------------------
print('-------------------------- Training Model --------------------------')
for steps in range(TRAIN_ITERATION):
    
    # Evaluate Loss
    if steps % EVAL_ITERATION == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Getting a mini-batch of data for training
    xb, yb = get_batch(split = 'train', batch_size= BATCH_SIZE, block_size= BLOCK_SIZE, device=DEVICE)
    
    # Forward Pass
    logits , loss = model(xb, yb)
    
    # Setting the gradient to zeros
    optimizer.zero_grad()
    
    # Backward Pass
    loss.backward()
    
    # Updating the gradients
    optimizer.step()

print('-------------------------- Making new context using model --------------------------')

# ------------------------- Generating sample from model -------------------------------
context = torch.zeros((1,1), dtype=torch.long).to(DEVICE)
print(decode(model.generate(context=context, max_length=500)[0].tolist()))



