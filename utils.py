import numpy as np
from torch.utils import data

encode = lambda bag, s: [bag.index(l) for l in s]
decode = lambda bag, ids: "".join([bag[id] for id in ids])

class ShDataset(data.Dataset):

  def __init__(self, data, bag, T):
    self.data = data
    self.T = T
    self.bag = bag

  def __len__(self):
    return len(self.data) - self.T - 1

  def __getitem__(self, idx):
    return (np.array(encode(self.bag, self.data[idx:idx+self.T]), dtype=np.float32).T,
           self.bag.index(self.data[idx+self.T]))