from yacs.config import CfgNode as CN

cfg = CN({
  'exp_name': '',
  'logdir': '/data/local/cc1547/projects/dreamer/dreamer_pytorch/CC/dreamer_and_trans_dreamer_exp5',
  'resume': True,
  'resume_ckpt': '',
  'debug': False,
  'seed': 0,
  'run_id': 'run_0',
  'total_steps': 1e7,
  'arch':{
    'use_pcont': True,
    'mem_size': 100000,
    'prefill': 50000,
    'H': 15,
    'world_model': {
      'reward_layer': 0,
      'q_emb_action': False,
      'RSSM': {
        'act': 'elu',
        'weight_init': 'xavier',
        'stoch_size': 32,
        'stoch_discrete': 32,
        'deter_size': 2048,
        'hidden_size': 2048,
        'rnn_type': 'LayerNormGRUV2',
        'ST': True,
        'post_no_deter': False,
      },
      'transformer': {
        'max_time': 2000,
        'num_heads': 8,
        'd_model': 600,
        'd_inner': 64,
        'd_ff_inner': 512,
        'dropout': 0.1,
        'dropatt': 0.1,
        'activation': 'relu',
        'pos_enc': 'temporal',
        'embedding_type': 'linear',
        'n_layers': 8,
        'pre_lnorm': True,
        'deter_type': 'concat_o',
        'gating': False,
      },
      'q_transformer': {
        'max_time': 2000,
        'num_heads': 8,
        'd_model': 600,
        'd_inner': 64,
        'd_ff_inner': 512,
        'dropout': 0.1,
        'dropatt': 0.1,
        'activation': 'relu',
        'pos_enc': 'temporal',
        'embedding_type': 'linear',
        'n_layers': 4,
        'pre_lnorm': True,
        'deter_type': 'concat_o',
        'q_emb_action': False,
        'gating': False,
      },
      'reward': {
        'num_units': 400,
        'act': 'elu',
        'dist': 'normal',
        'layers': 4,
      },
      'pcont': {
        'num_units': 400,
        'dist': 'binary',
        'act': 'elu',
        'layers': 4,
      },
    },
    'actor': {
      'num_units': 400,
      'act': 'elu',
      'init_std': 5.0,
      'dist': 'onehot',
      'layers': 4,
      'actor_loss_type': 'reinforce',
    },
    'value': {
      'num_units': 400,
      'act': 'elu',
      'dist': 'normal',
      'layers': 4,
    },

    'decoder': {
      'dec_type': 'conv',
    }
  },
  'loss': {
    'pcont_scale': 5.,
    'kl_scale': 0.1,
    'free_nats': 0.,
    'kl_balance': 0.8,
    'ent_scale': 3e-3,
  },

  'env':{
    'action_size': 17,
    'name': 'crafter_crafter',
    'action_repeat': 4,
    'max_steps': 1000,
    'life_done': False,
    'precision': 16,
    'time_limit': 108000,
    'grayscale': False,
  },
  'rl': {
    'discount': 0.999,
    'lambda_': 0.95,
    'expl_amount': 0.0,
    'expl_decay': 200000.0,
    'expl_min': 0.0,
    'expl_type': 'epsilon_greedy',
    'r_transform': 'tanh',
  },
  'data':{
    'datadir': '/data/local/cc1547/projects/dreamer/dreamer_pytorch/CC/dreamer_and_trans_dreamer_exp5',
  },
  'train': {
    'batch_length': 50,
    'batch_size': 50,
    'train_steps': 100,
    'train_every': 16,
    'print_every_step': 2000,
    'log_every_step': 1e3,
    'checkpoint_every_step': 1e4,
    'eval_every_step': 1e5,
    'n_sample': 10,
    'imag_last_T': False,
  },
  'optimize': {
    'model_lr': 1e-4,
    'value_lr': 1e-4,
    'actor_lr': 1e-4,
    'optimizer': 'adamW',
    'grad_clip': 100.,
    'weight_decay': 1e-6,
    'eps': 1e-5,
    'reward_scale': 1.,
    'discount_scale': 5.,
  },

  'checkpoint': {
    'checkpoint_dir': '/data/local/cc1547/projects/dreamer/dreamer_pytorch/CC/dreamer_and_trans_dreamer_exp5',
    'max_num': 10,
  },
})