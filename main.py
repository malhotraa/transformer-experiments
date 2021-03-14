from torch.nn import MSELoss

from exploration.dataset_exploration import build_dataset
from transformer.pytorch.model import make_model

num_samples = 10
num_epochs = 5
dmodel = 64

def run_epoch(dataset, model, loss_func):
    loss = 0
    for i in range(num_samples):
        x, y = dataset[i]
        out = model.forward(x, y, src_mask=None, tgt_mask=None)
        loss += loss_func.forward(y, out)
    return loss

train_dataset = build_dataset(block_size=dmodel)
model = make_model(train_dataset.vocab_size, train_dataset.vocab_size, d_model=dmodel)
loss_func = MSELoss()

for epoch in range(num_epochs):
    model.train()
    loss = run_epoch(train_dataset, model, loss_func)
    print(f"Loss epoch {epoch}: {loss}")

model.eval()
