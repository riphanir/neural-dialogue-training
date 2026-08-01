"""
trainer.py
The real training step: takes an "ideal" text (Groq's corrected reply) and
runs actual backpropagation on the tiny model's weights (imitation learning
via next-character prediction loss).
"""

import torch
import torch.nn.functional as F
from tiny_transformer import encode, PAD_ID, MAX_SEQ_LEN


def train_on_text(model, target_text, epochs=3, lr=3e-4):
    """
    Trains the model to predict target_text character-by-character
    (standard LM loss). This is the mechanism that actually changes the
    weights (not just overwriting text).
    Returns the first and last loss values so we can track improvement.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    ids = encode(target_text, max_len=MAX_SEQ_LEN)
    if len(ids) < 3:
        return None, None

    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)  # (1, seq_len, vocab_size)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=PAD_ID,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return losses[0], losses[-1]
  
