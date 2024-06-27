import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# matn yuklash
with open('shaytonat 1-3.txt', 'r') as f:
  text = f.read()
bag = sorted(list(set(text)))
n_bag = len(bag)
print(f'Beliglar soni: {n_bag}')

encode = lambda s: [bag.index(l) for l in s]
decode = lambda ids: "".join([bag[id] for id in ids])

# Matni ikki qismga ajratish
n_data = len(text)
val_size = 0.1
n_train = int((1 - val_size) * n_data)
n_val = n_data - n_train

train_data = torch.tensor(encode(text[:n_train]), dtype=torch.long)
val_data = torch.tensor(encode(text[n_train:]), dtype=torch.long)
print("O'rgatuvchida: ", n_train)
print("Sinovda: ", n_val)

batch_size = 256
block_size = 192
vocab_size = n_bag

embed_dim = 192
num_heads = 16
num_blocks = 5
dropout = 0.4
lr = 1e-3

class Dataset(data.Dataset):

    def __init__(self,
                 data_idx):
        super().__init__()
        self.size = data_idx.shape[0] - block_size - 1
        self.data = data_idx
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (self.data[idx:idx+block_size],
                self.data[idx+1:idx+block_size+1])

train_loader = data.DataLoader(
    dataset=Dataset(train_data),
    batch_size=batch_size,
    num_workers=5,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

val_loader = data.DataLoader(
    dataset=Dataset(val_data),
    batch_size=batch_size,
    num_workers=5,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

class Head(nn.Module):

    def __init__(self,
                 embed_dim,
                 head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_dim, head_size) # (B, T, head_size)
        self.key = nn.Linear(embed_dim, head_size)   # (B, T, head_size)
        self.value = nn.Linear(embed_dim, head_size) # (B, T, head_size)

        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
    
    def forward(self, x):
        T = x.shape[1]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1)
        wei = wei / (self.head_size ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v
        return out


class MultiHead(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 head_size,
                 num_heads,
                 dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_dim, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout):
        super().__init__()
        
        self.norm_layer_1 = nn.LayerNorm(embed_dim)
        self.multi_head = MultiHead(embed_dim, 
                                    embed_dim // num_heads, 
                                    num_heads,
                                    dropout)
        self.norm_layer_2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.multi_head(self.norm_layer_1(x))
        x = x + self.ff(self.norm_layer_2(x))
        return x


class BigramLM(pl.LightningModule):

    def __init__(self,
                 embed_dim,
                num_blocks,
                num_heads,
                dropout,
                lr):
        super().__init__()
        # self.hparams
        self.save_hyperparameters()

        self.token_embedding_table = nn.Embedding(vocab_size, 
                                                  embed_dim)
        self.postion_embedding_table = nn.Embedding(block_size, 
                                                  embed_dim)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, 
                  num_heads,
                  dropout) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, xb, yb=None):
        B, T = xb.shape
        # xb -> (batch_size, block_size) => (4, 8)
        # yb -> (batch_size, block_size) => (4, 8)
        # (batch_size, block_size, n_emb)
        # (4, 8, 32)
        token_emb = self.token_embedding_table(xb)
        postion_emb = self.postion_embedding_table(torch.arange(0, T, device=device))
        x = self.blocks(token_emb + postion_emb)
        # (4, 8, 93)
        logits = self.fc(x)
        
        if yb is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            yb = yb.view(B*T)
            loss = F.cross_entropy(logits, yb)
        else:
            loss = None

        return logits, loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                                self.hparams.lr)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1, 2, 3, 4], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        xb, yb = batch
        logits, loss = self.forward(xb, yb)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        xb, yb = batch
        logits, loss = self.forward(xb, yb)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('val_loss', loss)
    
    def on_train_epoch_end(self) -> None:
        sup_res = super().on_test_epoch_end()
        # text generating
        idx = torch.tensor([encode("\n")], 
                    dtype=torch.long, 
                    device=device)
        
        print('Generated text:')
        self.generate(idx, max_new_token=5000)

        return sup_res

    def test_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        xb, yb = batch
        logits, loss = self.forward(xb, yb)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('test_loss', loss)

    @torch.no_grad
    def generate(self, idx, max_new_token):
        self.eval()
        for _ in range(max_new_token):
            # (batch_size, block_size, vocab_size)
            # (1, 8, 93)
            idx = idx[:, -block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            # idx_next = torch.argmax(probs, keepdim=True)
            
            print(decode(idx_next[0]), end='')
            
            idx = torch.cat((idx, idx_next), dim=1)
        print()
        
        self.train()


model = BigramLM.load_from_checkpoint('checkpoints/llm/lightning_logs/version_12/checkpoints/epoch=5-step=43308.ckpt')
model.train()

idx = torch.tensor([encode(". ")], 
                    dtype=torch.long, 
                    device=device)
model.generate(idx, max_new_token=1000)