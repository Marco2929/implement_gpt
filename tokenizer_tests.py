import torch
import tiktoken

torch.manual_seed(42)

# Load the text (for example, from tinyshakespeare dataset)
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# --- Character-level Tokenization (Normal Version) ---
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

# --- tiktoken Tokenization ---
enc = tiktoken.get_encoding("o200k_base")  # Tokenize the entire text as a single tokenized block

data_tiktoken = torch.tensor(enc.encode(text), dtype=torch.long)

pass
