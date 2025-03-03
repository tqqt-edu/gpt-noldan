import os
import numpy as np
import glob
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from tokenizers import Tokenizer

from llm import BigramLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

tokenizer = Tokenizer.from_file("./lexuz-token.json")
print(f'Tokenlar soni: {tokenizer.get_vocab_size()}')

encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)

batch_size = 32
lr = 1e-5

vocab_size = tokenizer.get_vocab_size()
block_size = 512
embed_dim = 512
num_heads = 16
num_blocks = 10
dropout = 0.3

class DataLoader:

    def __init__(self,
                 block_size,
                 batch_size,
                 data_dir,
                 tokenizer,
                 drop_last=True,
                 shuffle=True):
        
        self.block_size = block_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        files = list(glob.glob(f'{data_dir}/*.txt'))
        self.tokens = []
        for file_path in files:
            with open(file_path, 'r') as f:
                text = ' '.join(f.readlines())
                self.tokens.append(torch.tensor(tokenizer.encode(text).ids, 
                                                dtype=torch.long))
        self.tokens = torch.cat(self.tokens)
    
    def __len__(self):
        return (self.tokens.shape[0] - self.block_size - 1) // self.batch_size + 0 if self.drop_last  else 1

    def __iter__(self):
        length = self.tokens.shape[0] - self.block_size - 1
        indices = torch.arange(length)
        if self.shuffle:
            indices = torch.randperm(length)
        
        for idx in range(len(self)):
            xb = []
            yb = []
            for st_idx in indices[idx*self.batch_size:(idx+1)*self.batch_size]:
                xb.append(self.tokens[st_idx:st_idx+self.block_size])
                yb.append(self.tokens[st_idx+1:st_idx+self.block_size+1])

            xb = torch.stack(xb)
            yb = torch.stack(yb)
            
            yield xb, yb


train_data_dir = '/home/tqqt/Documents/AI/projects/gpts/training/data/lexuz/train'
val_data_dir = '/home/tqqt/Documents/AI/projects/gpts/training/data/lexuz/val'

train_loader = DataLoader(block_size=block_size,
                        batch_size=batch_size,
                        data_dir=train_data_dir,
                        tokenizer=tokenizer)

val_loader = DataLoader(block_size=block_size,
                        batch_size=batch_size,
                        data_dir=val_data_dir,
                        tokenizer=tokenizer,
                        drop_last=False,
                        shuffle=False)

def train_model(save_name='v11'):
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
                         max_epochs=1,                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, 
                                                    mode="min", 
                                                    monitor="val_loss",
                                                    every_n_train_steps=1000),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch"),
                                    EarlyStopping(monitor="val_loss", 
                                                  mode="min",
                                                  patience=10)],                                           # Log learning rate every epoch
                         enable_progress_bar=True,
                         val_check_interval=1000)                                                           # Set to False if you do not want a progress bar
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECK_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = BigramLM.load_from_checkpoint(pretrained_filename,
                                              token_encode=encode,
                                              token_decode=decode) # Automatically loads the model with the saved hyperparameters
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
        model = BigramLM.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                              token_encode=encode,
                                              token_decode=decode) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    result = {"val_loss": val_result[0]["test_loss"]}

    return model, result

model, result = train_model()