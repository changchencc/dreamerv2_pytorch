from .minigrid_env import GymGridEnv, OneHotAction, TimeLimit, Collect, RewardObs, NormalizeActions
from .atari_env import Atari
from .dmc import DeepMindControl
from .crafter import Crafter
from .unity_maze_env import IMaze3D
from .unity_maze_env_multy import IMaze3D16Area
from .tools import count_episodes, save_episodes, video_summary
import pathlib
import pdb
import json

def count_steps(datadir, cfg):
  return tools.count_episodes(datadir)[1]

def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, steps = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.env.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'{prefix}/episodes', episodes)]
  step = count_steps(datadir, config)
  env_step = step * config.env.action_repeat
  with (pathlib.Path(config.logdir) / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', env_step)] + metrics)) + '\n')
  [writer.add_scalar('sim/' + k, v, env_step) for k, v in metrics]
  if prefix == 'test':
    tools.video_summary(writer, f'sim/{prefix}/video', episode['image'][None, :1000], env_step)
    if 'top_view' in episode:
      tools.video_summary(writer, f'sim/{prefix}/topView-video', episode['top_view'][None, :1000], env_step)

  if prefix == 'train':
    tools.video_summary(writer, f'sim/{prefix}/video', episode['image'][None, :1000], env_step)
    if 'top_view' in episode:
      tools.video_summary(writer, f'sim/{prefix}/topView-video', episode['top_view'][None, :1000], env_step)

  if 'episode_done' in episode:
    num_episodes = max(1, sum(episode['episode_done']))
    num_suceed = sum(episode['succeed'])
    succeed_ratio = float(num_suceed / num_episodes)
    writer.add_scalar('sim/succeed_ratio', succeed_ratio, env_step)
    writer.add_scalar('sim/num_episodes', num_episodes, env_step)
  writer.flush()

def make_env(cfg, writer, prefix, datadir, store, seed=0):

  suite, task = cfg.env.name.split('_', 1)

  if suite == 'dmc':
    env = DeepMindControl(task, cfg.env.action_repeat)
    env = wrappers.NormalizeActions(env)

  elif suite == 'atari':
    env = Atari(
        task, cfg.env.action_repeat, (64, 64), grayscale=cfg.env.grayscale,
        life_done=False, sticky_actions=True, seed=seed, all_actions=cfg.env.all_actions)
    env = OneHotAction(env)

  elif suite == 'minigrid':
    env = GymGridEnv(
      task, cfg.env.action_repeat, max_steps=cfg.env.max_steps, life_done=cfg.env.life_done)
    env = OneHotAction(env)

  elif suite == 'crafter':
    env = Crafter(task, (64, 64), seed)
    env = OneHotAction(env)

  elif suite == 'unity':
    id = 0 if prefix == 'train' else 1
    env = IMaze3D(task, cfg.env.env_file, seed+id, cfg.env.action_repeat, cfg.env.action_size, cfg.env.top_view,
                  (64, 64), grayscale=False, seed=seed + id, life_done=cfg.env.life_done)
    if cfg.arch.actor.dist == 'onehot':
      env = OneHotAction(env)
    else:
      env = NormalizeActions(env)

  elif suite == 'unity16':
    id = 0 if prefix == 'train' else 1

    callbacks = []
    if store:
      callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(
      lambda ep: summarize_episode(ep, cfg, datadir, writer, prefix))

    env = IMaze3D16Area(task, cfg.env.env_file, seed+id, cfg.env.action_size, callbacks, cfg.env.max_steps, cfg.env.num_area,
                  (64, 64), seed=seed + id, life_done=cfg.env.life_done)
  else:
    raise NotImplementedError(suite)

  if suite != 'unity16':
    env = TimeLimit(env, cfg.env.time_limit, cfg.env.time_penalty)

    callbacks = []
    if store:
      callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(
        lambda ep: summarize_episode(ep, cfg, datadir, writer, prefix))
    env = Collect(env, callbacks, cfg.env.precision)
    env = RewardObs(env)

  return env
