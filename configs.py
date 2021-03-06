from yacs.config import CfgNode as CN

cfg = CN(
    {
        "exp_name": "",
        "logdir": "/data/cc1547/projects/dreamer/dreamer_pytorch/CC_V3/dreamer_V2",
        "resume": True,
        "resume_ckpt": "",
        "debug": False,
        "seed": 0,
        "run_id": "run_0",
        "total_steps": 1e7,
        "model": "dreamer",
        "slow_update_step": 100,
        "deter_rollout": False,
        "arch": {
            "local_state": False,
            "use_pcont": True,
            "mem_size": 100000,
            "prefill": 50000,
            "H": 15,
            "q_trans": False,
            "patch_size": 32,
            "actor": {
                "num_units": 400,
                "act": "elu",
                "init_std": 5.0,
                "dist": "onehot",
                "layers": 4,
                "aggregator": "none",
                "actor_loss_type": "both",
            },
            "value": {"num_units": 400, "act": "elu", "dist": "normal", "layers": 4,},
            "world_model": {
                "reward_layer": 4,
                "q_emb_action": False,
                "act_after_emb": False,
                "rec_sigma": 1.0,
                "anneal_world_model": False,
                "discrete_type": "discrete",
                "std_type": "sigmoid2",
                "temp_start": 2.0,
                "temp_end": 0.001,
                "temp_decay_steps": 1e6,
                "transformer": {
                    "max_time": 2000,
                    "num_heads": 8,
                    "d_model": 600,
                    "d_inner": 64,
                    "d_ff_inner": 1024,
                    "dropout": 0.1,
                    "dropatt": 0.1,
                    "activation": "relu",
                    "pos_enc": "temporal",
                    "embedding_type": "linear",
                    "n_layers": 2,
                    "pre_lnorm": True,
                    "deter_type": "concat_o",
                    "gating": False,
                    "last_ln": False,
                    "enc_pos": False,
                    "warm_up": False,
                },
                "q_transformer": {
                    "max_time": 2000,
                    "num_heads": 8,
                    "d_model": 600,
                    "d_inner": 64,
                    "d_ff_inner": 1024,
                    "dropout": 0.1,
                    "dropatt": 0.1,
                    "activation": "relu",
                    "pos_enc": "temporal",
                    "embedding_type": "linear",
                    "n_layers": 2,
                    "pre_lnorm": True,
                    "deter_type": "concat_o",
                    "q_emb_action": False,
                    "gating": False,
                    "last_ln": False,
                },
                "gtrxl": {
                    "d_model": 512,
                    "d_inner": 64,
                    "d_ff_inner": 512,
                    "n_head": 8,
                    "mem_len": 16,
                    "dropout": 0.1,
                    "dropatt": 0.1,
                    "pre_lnorm": True,
                    "n_layers": 4,
                    "gating": True,
                    "deter_type": "concat_o",
                },
                "RSSM": {
                    "act": "elu",
                    "weight_init": "xavier",
                    "stoch_size": 32,
                    "stoch_discrete": 32,
                    "deter_size": 600,
                    "hidden_size": 600,
                    "rnn_type": "LayerNormGRU",
                    "ST": True,
                    "post_no_deter": False,
                },
                "reward": {
                    "num_units": 400,
                    "act": "elu",
                    "dist": "normal",
                    "layers": 4,
                },
                "pcont": {
                    "num_units": 400,
                    "dist": "binary",
                    "act": "elu",
                    "layers": 4,
                },
            },
            "static_wm": {
                "recon_sigma": 0.2,
                "G": 4,
                "vq_num_embeddings": 32,
                "vq_embedding_dim": 64,
                "vq_commitment_beta": 0.25,
                "decay": 0.99,
                "epsilon": 1e-5,
                "enc_v": "v2",
                "dec_v": "v3",
            },
            "decoder": {"dec_type": "conv",},
        },
        "loss": {
            "pcont_scale": 5.0,
            "kl_scale": 0.1,
            "free_nats": 0.0,
            "kl_balance": 0.8,
            "ent_scale": 1e-3,
        },
        "env": {
            "action_size": 18,
            "name": "atari_boxing",
            "action_repeat": 4,
            "max_steps": 1000,
            "life_done": True,
            "precision": 16,
            "time_limit": 1000,
            "grayscale": False,
            "random_crop": True,
            "all_actions": True,
            "resize": True,
            "crop_repeat": 4,
            "env_file": "",
            "top_view": False,
            "time_penalty": False,
            "num_area": 16,
        },
        "rl": {
            "discount": 0.999,
            "lambda_": 0.95,
            "expl_amount": 0.4,
            "expl_decay": 200000.0,
            "expl_min": 0.1,
            "expl_type": "epsilon_greedy",
            "r_transform": "tanh",
        },
        "data": {
            "datadir": "/data/cc1547/projects/dreamer/dreamer_pytorch/CC_V3/dreamer_V2",
        },
        "train": {
            "batch_length": 50,
            "batch_size": 50,
            "train_steps": 50,
            "train_every": 16,
            "print_every_step": 1000,
            "log_every_step": 1e4,
            "checkpoint_every_step": 1e4,
            "eval_every_step": 1e5,
            "n_sample": 10,
            "imag_last_T": False,
            "log_grad": True,
        },
        "optimize": {
            "warmup_iter": 4e3 * 16,
            "exp_rate": 0.5,
            "decay_step": 1e5 * 16,
            "base_lr": 2e-4,
            "end_lr": 1e-4,
            "model_lr": 2e-4,
            "value_lr": 1e-4,
            "actor_lr": 1e-5,
            "optimizer": "adamW",
            "grad_clip": 100.0,
            "eps": 1e-5,
            "weight_decay": 1e-6,
            "reward_scale": 1.0,
            "discount_scale": 5.0,
        },
        "checkpoint": {
            "checkpoint_dir": "/data/cc1547/projects/dreamer/dreamer_pytorch/CC_V3/dreamer_V2",
            "max_num": 3,
        },
    }
)

