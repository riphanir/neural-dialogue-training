"""
tiny_transformer.py
A real, small character-level language model (~5M parameters).
Weights are randomly initialized on first run, then actually updated via
backpropagation on every training cycle, based on Groq's corrected replies.
"""

import os
import math
import string
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- vocabulary definition (English letters + digits + basic punctuation) ----
EXTRA = string.ascii_letters + string.digits + " \n.,!?:;-\"'()"
VOCAB = sorted(set(EXTRA))
VOCAB = ["<pad>", "<bos>", "<eos>", "<unk>"] + VOCAB
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

PAD_ID = STOI["<pad>"]
BOS_ID = STOI["<bos>"]
EOS_ID = STOI["<eos>"]
UNK_ID = STOI["<unk>"]

# ---- model config (tuned to land around ~5M parameters) ----
D_MODEL = 320
N_HEADS = 8
N_LAYERS = 6
D_FF = 768
MAX_SEQ_LEN = 512
DROPOUT = 0.1


def encode(text, max_len=MAX_SEQ_LEN):
    ids = [BOS_ID] + [STOI.get(ch, UNK_ID) for ch in text[: max_len - 2]] + [EOS_ID]
    return ids


def decode(ids):
    chars = []
    for i in ids:
        if i in (PAD_ID, BOS_ID):
            continue
        if i == EOS_ID:
            break
        chars.append(ITOS.get(i, ""))
    return "".join(chars)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TinyCharTransformer(nn.Module):
    """
    Decoder-only tiny transformer, character-level.
    ~5M parameters with these settings (D_MODEL=320, 6 layers, 8 heads).
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        # tie embedding and output head weights - saves parameters
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        # small-scale init (GPT-2 style) so training is stable from step one
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1).to(x.device)
        h = self.embed(x) * math.sqrt(D_MODEL)
        h = self.pos_enc(h)
        h = self.encoder(h, mask=mask)
        h = self.ln_f(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prompt_text, max_new_tokens=120, temperature=0.9, top_k=20):
        self.eval()
        ids = [BOS_ID] + [STOI.get(ch, UNK_ID) for ch in prompt_text[-100:]]
        x = torch.tensor([ids], dtype=torch.long)
        for _ in range(max_new_tokens):
            x_cond = x[:, -MAX_SEQ_LEN:]
            logits = self(x_cond)[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)
            if next_id.item() == EOS_ID:
                break
        return decode(x[0].tolist()[1:])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_or_init_model(weights_path):
    model = TinyCharTransformer()
    is_new = not os.path.exists(weights_path)
    if not is_new:
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    return model, is_new


def save_model(model, weights_path):
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)


if __name__ == "__main__":
    m = TinyCharTransformer()
    print(f"Parameter count: {count_parameters(m):,}")
