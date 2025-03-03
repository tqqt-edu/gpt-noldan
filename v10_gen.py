import os
import numpy as np
import glob

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

tokenizer = Tokenizer.from_file("./lexuz-token.json")
print(f'Tokenlar soni: {tokenizer.get_vocab_size()}')

encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)


model = BigramLM.load_from_checkpoint('/home/tqqt1/AI/teachings/online-courses/gpt_scratch/checkpoints/v10/lightning_logs/version_9/checkpoints/epoch=0-step=41000.ckpt',
                                      token_encode=encode,
                                      token_decode=decode)
model.train()

idx = torch.tensor([encode("\n")], 
                    dtype=torch.long, 
                    device=device)
model.generate(idx, max_new_token=1000)