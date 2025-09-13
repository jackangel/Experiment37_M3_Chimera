import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from tqdm import tqdm
import sys
import os
# No need to import torch.jit separately, it's part of torch

# --- 1. Configuration (Identical) ---
class Config:
    FILE_PATH = "input.txt"
    CONTEXT_SIZE = 1024
    EMBEDDING_DIM = 256
    NUM_LAYERS = 8
    BATCH_SIZE = 4
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    PREDICT_EVERY = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "geodesic_state_model_checkpoint.pth"

# --- 2. Data Preparation (Identical) ---
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.lower().split()
    vocab = sorted(list(set(tokens)))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    return tokens, vocab, word_to_idx, idx_to_word

class TextDataset(Dataset):
    def __init__(self, token_indices, context_size):
        self.token_indices = token_indices
        self.context_size = context_size

    def __len__(self):
        return len(self.token_indices) - self.context_size

    def __getitem__(self, idx):
        inputs = self.token_indices[idx : idx + self.context_size]
        targets = self.token_indices[idx + 1 : idx + self.context_size + 1]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# --- 3. The Geodesic State-Space Model (Correct JIT Application) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# --- MODIFICATION: The decorator is removed. This is now a standard nn.Module. ---
class GeodesicStateBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_layer = nn.Linear(embed_dim * 2, embed_dim * 2)
        self.candidate_layer = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        B, S, D = points.shape
        hidden_state = torch.zeros(B, D, device=points.device, dtype=points.dtype)
        
        outputs: list[torch.Tensor] = []
        for t in range(S):
            p_t = points[:, t, :]
            combined_input = torch.cat([p_t, hidden_state], dim=1)
            
            reset_gate, update_gate = self.gate_layer(combined_input).chunk(2, dim=1)
            reset_gate = torch.sigmoid(reset_gate)
            update_gate = torch.sigmoid(update_gate)
            
            candidate_input = torch.cat([p_t, reset_gate * hidden_state], dim=1)
            candidate_state = torch.tanh(self.candidate_layer(candidate_input))
            
            hidden_state = (1.0 - update_gate) * hidden_state + update_gate * candidate_state
            
            outputs.append(hidden_state)
            
        states = torch.stack(outputs, dim=1)
        displacement = states - points
        deformed_points = points + displacement
        return self.norm(deformed_points)


class GeodesicLanguageModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        self.positional_encoding = PositionalEncoding(config.EMBEDDING_DIM, config.CONTEXT_SIZE)
        
        # --- MODIFICATION: We apply torch.jit.script() to each INSTANCE of the block. ---
        layers = [
            GeodesicStateBlock(config.EMBEDDING_DIM) for _ in range(config.NUM_LAYERS)
        ]
        self.layers = nn.ModuleList([torch.jit.script(layer) for layer in layers])
        
        self.lm_head = nn.Linear(config.EMBEDDING_DIM, vocab_size)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx):
        x = self.token_embedding(idx)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits

# --- 4. Training and Generation (Identical) ---
def generate_text(model, start_prompt, max_len, word_to_idx, idx_to_word, config):
    model.eval()
    words = start_prompt.lower().split()
    context_tokens = [word_to_idx.get(w, 0) for w in words]
    generated_tokens = list(context_tokens)

    for _ in range(max_len):
        input_tokens = torch.tensor([generated_tokens[-config.CONTEXT_SIZE:]], dtype=torch.long).to(config.DEVICE)
        with torch.no_grad():
            logits = model(input_tokens)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1).item()
        generated_tokens.append(next_token_idx)
    
    model.train()
    return ' '.join([idx_to_word.get(i, '?') for i in generated_tokens])

def main():
    config = Config()
    print(f"Using device: {config.DEVICE}")

    tokens, vocab, word_to_idx, idx_to_word = load_data(config.FILE_PATH)
    vocab_size = len(vocab)
    token_indices = [word_to_idx[t] for t in tokens]
    
    dataset = TextDataset(token_indices, config.CONTEXT_SIZE)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    print(f"Vocabulary size: {vocab_size}")

    model = GeodesicLanguageModel(vocab_size, config).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"\n--- Checkpoint found at '{config.CHECKPOINT_PATH}'. Entering chat mode. ---")
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("\nType your prompt and press Enter. Type 'quit' or 'exit' to end.")
        while True:
            try:
                prompt = input("> ")
                if prompt.lower() in ['quit', 'exit']: break
                if not prompt.strip(): continue
                generated_text = generate_text(
                    model=model, start_prompt=prompt, max_len=100,
                    word_to_idx=word_to_idx, idx_to_word=idx_to_word, config=config
                )
                print(f"Model: {generated_text}\n")
            except KeyboardInterrupt:
                print("\nExiting chat mode.")
                break
        return

    print(f"Training examples: {len(dataset)}")
    print("--- No checkpoint found. Starting training. ---")

    step = 0
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        last_loss = 0
        for inputs, targets in pbar:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            last_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if (step + 1) % config.PREDICT_EVERY == 0:
                print(f"\n--- Step {step+1}: Generating text ---")
                prompt = "the king"
                generated = generate_text(model, prompt, max_len=30, 
                                          word_to_idx=word_to_idx, 
                                          idx_to_word=idx_to_word, 
                                          config=config)
                print(f"Prompt: '{prompt}'\nOutput: '{generated}'\n")
            
            step += 1
        
        print(f"--- End of Epoch {epoch+1}. Saving model checkpoint... ---")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': last_loss,
        }, config.CHECKPOINT_PATH)
        print(f"Checkpoint saved to '{config.CHECKPOINT_PATH}'")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print(f"\nERROR: Could not find the corpus file '{Config.FILE_PATH}'.")
        print("Please create this file and add some text to it before running.")
        sys.exit(1)