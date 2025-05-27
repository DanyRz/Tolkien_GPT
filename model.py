import torch
import torch.nn as nn
from torch.nn import functional

import hyperparameters
import encoder
import attention


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(encoder.vocabulary_size, hyperparameters.n_embed)
        self.position_embedding_table = nn.Embedding(hyperparameters.block_size, hyperparameters.n_embed)
        self.blocks = nn.Sequential(*[attention.Block(hyperparameters.n_embed, n_head=hyperparameters.n_head) for _ in
                                      range(hyperparameters.n_layer)])
        self.ln_f = nn.LayerNorm(hyperparameters.n_embed)
        self.lm_head = nn.Linear(hyperparameters.n_embed, encoder.vocabulary_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, time_step = idx.shape

        token_embed = self.token_embedding_table(idx)
        position_embed = self.position_embedding_table(torch.arange(time_step, device=hyperparameters.device))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, time_step, channels = logits.shape
            logits = logits.view(batch * time_step, channels)
            targets = targets.view(batch * time_step)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -hyperparameters.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = functional.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
