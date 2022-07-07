import torch
from .dreamer import Dreamer
from .dreamer_vqvae import DreamerVQVAE
from .dreamer_transformer import DreamerTransformer
from .dreamer_transformer_multienvs import DreamerTransformerMultienv
from .dreamer_gtrxl import DreamerGTrxl
from .dreamer_transformer_partial import DreamerTransformerPartial
from .dreamer_partial import DreamerPartial

def get_model(cfg, device, seed=0):

  torch.manual_seed(seed=seed)
  if cfg.model == 'dreamer':
    model = Dreamer(cfg)

  if cfg.model == 'dreamer_partial':
    model = DreamerPartial(cfg)

  if cfg.model == 'dreamer_vqvae':
    model = DreamerVQVAE(cfg)


  if cfg.model == 'dreamer_transformer':
    model = DreamerTransformer(cfg)


  if cfg.model == 'dreamer_transformer_multienv':
    model = DreamerTransformerMultienv(cfg)

  if cfg.model == 'dreamer_transformer_partial':
    model = DreamerTransformerPartial(cfg)


  if cfg.model == 'dreamer_gtrxl':
    model = DreamerGTrxl(cfg)


  return model

