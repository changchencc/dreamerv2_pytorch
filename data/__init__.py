import pdb

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader, get_worker_info
from model.utils import spatial_transformer
from utils import crop_obs
import os
import pathlib
import numpy as np
import glob
import torch

class EnvIterDataset(IterableDataset):
  def __init__(self, data_dir, train_steps, batch_length, seed=0):
    self.data_dir = data_dir
    self.batch_length = batch_length
    self.train_steps = train_steps
    self.seed = seed

  def load_episodes(self, balance=False):
    directory = pathlib.Path(self.data_dir).expanduser()
    worker_info = get_worker_info()
    random = np.random.RandomState((self.seed + worker_info.seed) % (1 << 32))
    cache = {}
    while True:
      for filename in directory.glob('*.npz'):
        if filename not in cache:
          cache[filename] = filename

      keys = list(cache.keys())
      indices = random.choice(len(keys), self.train_steps)
      # print(f'indices: {indices}')
      for index in indices:
        filename = cache[keys[index]]

        try:
          with open(filename, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
          print(f'Could not load episode: {e}')
          continue

        if self.batch_length:
          total = len(next(iter(episode.values())))
          available = total - self.batch_length
          if available < 1:
            print(f'Skipped short episode of length {available}.')
            continue
          if balance:
            index = min(random.randint(0, total), available)
          else:
            index = int(random.randint(0, available + 1))
            # index = available
          episode = {k: v[index: index + self.batch_length] for k, v in episode.items()}
        yield episode

  def __iter__(self):
    return self.load_episodes()

class PartialViewEnvIterDataset(IterableDataset):
  def __init__(self, data_dir, train_steps, batch_length, patch_size, random_crop, resize, seed=0, start_left_bottom=False):
    self.data_dir = data_dir
    self.batch_length = batch_length
    self.train_steps = train_steps
    self.random_crop = random_crop
    self.patch_size = patch_size
    self.resize = resize
    self.seed = seed
    self.start_left_bottom = start_left_bottom
    if random_crop:
      self.last_idx = None
    else:
      self.last_idx = 0

  def load_episodes(self, balance=False):
    directory = pathlib.Path(self.data_dir).expanduser()
    worker_info = get_worker_info()
    random = np.random.RandomState((self.seed + worker_info.seed) % (1 << 32))
    cache = {}
    while True:
      for filename in directory.glob('*.npz'):
        if filename not in cache:
          cache[filename] = filename

      keys = list(cache.keys())
      indices = random.choice(len(keys), self.train_steps)
      # print(f'indices: {indices}')
      for index in indices:
        filename = cache[keys[index]]

        try:
          with open(filename, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
          print(f'Could not load episode: {e}')
          self.seed = self.seed + 1
          random = np.random.RandomState((self.seed + worker_info.seed) % (1 << 32))
          continue

        if self.batch_length:
          total = len(next(iter(episode.values())))
          available = total - self.batch_length
          if available < 1:
            print(f'Skipped short episode of length {available}.')
            continue
          if balance:
            index = min(random.randint(0, total), available)
          else:
            index = int(random.randint(0, available+1))
          episode = {k: v[index: index + self.batch_length] for k, v in episode.items()}

          partial_o, pos_c, last_idx = crop_obs(episode['image'], self.patch_size, self.random_crop, self.last_idx, resize=self.resize)
          if self.start_left_bottom:
            if self.random_crop:
              self.last_idx = None
            else:
              self.last_idx = 0
          else:
            self.last_idx = last_idx
          episode['partial_image'] = partial_o
          episode['pos_c'] = pos_c
          episode['last_idx'] = last_idx
        yield episode

  def __iter__(self):
    return self.load_episodes()



