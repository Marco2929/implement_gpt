import importlib.util
import os

import torch

from get_tokenizer import MyTokenizer
from gpt import GPTLanguageModel
from utils import TokenizerType

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


def train_val_split(data, split=0.9):
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data

tokenizer = MyTokenizer(tokenizer_name = TokenizerType.TIKTOKEN)

vocab_size, data  = tokenizer.encode(text=text)

train_data, val_data = train_val_split(data=data)

def load_config(config_file):
    """Dynamically load configuration from the given Python file."""
    spec = importlib.util.spec_from_file_location("base_config", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Path to the configuration file
config_path = os.path.join(os.getcwd(), "configs", "base_config.py")

# Load the configuration
config = load_config(config_path)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(config=config, vocab_size=vocab_size)
m = model.to(device)

print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

for iter in range(config.max_iters):

    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))