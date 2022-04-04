import torch
from torch.utils.data import Dataset
import itertools
import numpy as np

def get_arithmetic_func(name):
    return {
        '+': lambda x,y,mod: (x, y, (x + y) % mod),
        '-': lambda x,y,mod: (x, y, (x - y) % mod),
        '/': lambda x,y,mod: ((x * y) % mod, y, x),
        '/-': lambda x,y,mod: (x, y, (x // y) % mod if y % 2 == 1 else (x - y) % mod),
        'x2+y2': lambda x,y,mod: (x, y, (x ** 2 + y ** 2) % mod),
        'x2+xy+y2': lambda x,y,mod: (x, y, (x ** 2 + x * y + y ** 2) % mod),
        'x2+xy+y2+x': lambda x,y,mod: (x, y, (x ** 2 + x * y + y ** 2 + x) % mod),
        'x3+xy': lambda x,y,mod: (x, y, (x ** 3 + x * y) % mod),
        'x3+xy+2y': lambda x,y,mod: (x, y, (x ** 3 + x * y ** 2 + y) % mod),
        's5': lambda x,y,mod: (x, y, y[x])
    }[name]

class Vocab:
  def __init__(self, tokens):
    self.stoi = {t:idx for idx,t in enumerate(sorted(tokens))}
    self.itos = {idx:t for idx,t in enumerate(sorted(tokens))}

  def encode(self, obj):
    return torch.tensor([self.stoi[str(s)] for s in obj])

  def decode(self, obj):
    return ' '.join([self.itos[i.item()] for i in obj])

class ArithmeticDataset(Dataset):
  def __init__(self, operator, mod=97):
    self.data, tokens = self.generate_data(operator, mod)
    self.vocab = Vocab(tokens)

  def __getitem__(self, index):
    return self.vocab.encode(self.data[index])

  def __len__(self):
    return len(self.data)

  def decode(self, item):
    return self.vocab.decode(item)

  @staticmethod
  def generate_data(operator, mod):
    function = get_arithmetic_func(operator)
    result = []
    tokens = {'<sos>', '=', str(operator)}

    if operator == '/':
      tuples = itertools.product(range(mod), range(1, mod))
    elif operator == 's5':
      elems = map(np.array, itertools.permutations(list(range(5))))
      tuples = itertools.product(elems, repeat=2)
    else:
      tuples = itertools.product(range(mod), repeat=2)

    for x, y in tuples:
      x, y, res = function(x, y, mod)
      result.append(['<sos>', x, operator, y, '=', res])
      tokens.update({str(x), str(y), str(res)})
    
    return result, tokens