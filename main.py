import torch
from torch.nn import MSELoss

from exploration.dataset_exploration import build_dataset, build_synthetic_dataset
from transformer.pytorch.model import make_model

batch_size = 10
num_epochs = 50
block_size = 8
dmodel = 64

def run_epoch(dataset, model, loss_func):
    x, y = dataset.batch(batch_size)
    out = model.forward(x, y, src_mask=None, tgt_mask=None)
    prob = model.generator(out)
    preds = torch.argmax(prob, dim=-1)
    loss = loss_func.forward(y.float(), preds.float())
    return loss

def greedy_decode(model):
    pass

# train_dataset = build_synthetic_dataset(block_size, num_samples) 
train_dataset = build_dataset(block_size=block_size)

model = make_model(train_dataset.vocab_size, train_dataset.vocab_size, d_model=dmodel)
loss_func = MSELoss()

for epoch in range(num_epochs):
    model.eval()
    loss = run_epoch(train_dataset, model, loss_func)
    print(f"Loss epoch {epoch}: {loss}")

model.eval()
