import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from tokenizers import Tokenizer

from llm import Dataset, BigramLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# matn (train) yuklash
with open('shaytonat 1-3-train.txt', 'r') as f:
  train_text = f.read()

# matn (val) yuklash
with open('shaytonat 1-3-val.txt', 'r') as f:
  val_text = f.read()

tokenizer = Tokenizer.from_file("./shaytonat-token.json")
print(f'Tokenlar soni: {tokenizer.get_vocab_size()}')

encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

print("Tokenlar o'rgatuvchida: ", train_data.shape)
print("Tokenlar sinovda: ", val_data.shape)


batch_size = 64
lr = 1e-3

vocab_size = tokenizer.get_vocab_size()
block_size = 192
embed_dim = 192
num_heads = 16
num_blocks = 4
dropout = 0.4

train_loader = data.DataLoader(
    dataset=Dataset(train_data, block_size),
    batch_size=batch_size,
    num_workers=5,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

val_loader = data.DataLoader(
    dataset=Dataset(val_data, block_size),
    batch_size=batch_size,
    num_workers=5,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

def train_model(save_name='v9'):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    CHECK_PATH = './checkpoints'
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECK_PATH, save_name),                          # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=1,                                                                          # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=30,                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, 
                                                    mode="min", 
                                                    monitor="val_loss"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch"),
                                    EarlyStopping(monitor="val_loss", 
                                                  mode="min",
                                                  patience=3)],                                           # Log learning rate every epoch
                         enable_progress_bar=True)                                                           # Set to False if you do not want a progress bar
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECK_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = BigramLM.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        model = BigramLM(embed_dim,
                num_blocks,
                num_heads,
                dropout,
                lr,
                block_size,
                vocab_size,
                encode,
                decode)
        trainer.fit(model, train_loader, val_loader)
        model = BigramLM.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    result = {"val_loss": val_result[0]["test_loss"]}

    return model, result

model, result = train_model()