import torch

import hyperparameters
import encoder
from model import Model

torch.manual_seed(679)
data = torch.tensor(encoder.encode(encoder.raw_data), dtype=torch.long)
split_point = int(0.85*len(data))
train_set = data[:split_point]
validation_set = data[split_point:]


def get_batch(split):
    work_data = train_set if split == 'train' else validation_set
    ix = torch.randint(len(work_data) - hyperparameters.block_size, (hyperparameters.batch_size,))
    data_input = torch.stack([work_data[i:i+hyperparameters.block_size] for i in ix])
    data_target = torch.stack([work_data[i+1:i+hyperparameters.block_size+1] for i in ix])
    data_input, data_target = data_input.to(hyperparameters.device), data_target.to(hyperparameters.device)
    return data_input, data_target


@torch.no_grad()
def estimate_loss():
    result = {}
    model.eval()
    for split in ['train', 'val']:
        current_losses = torch.zeros(hyperparameters.eval_iters)
        for k in range(hyperparameters.eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            current_losses[k] = loss.item()
        result[split] = current_losses.mean()
    model.train()
    return result


model = Model()
work_model = model.to(hyperparameters.device)
print('Number of parameters: ' + str(sum(p.numel() for p in work_model.parameters())))

optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.learning_rate)

for i in range(hyperparameters.max_iters):

    if i % hyperparameters.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    x_batch, y_batch = get_batch('train')
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=hyperparameters.device)
print(encoder.decode(work_model.generate(context, max_tokens=500)[0].tolist()))
