import sys

def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['xlarge'])
  # config = config.update({
  #     'logdir': '~/logdir/run1',
  #     'run.train_ratio': 64,
  #     'run.log_every': 30,  # Seconds
  #     'batch_size': 16,
  #     'jax.prealloc': False,
  #   #   'encoder.mlp_keys': '$^',
  #   #   'decoder.mlp_keys': '$^',
  #   #   'encoder.cnn_keys': 'image',
  #   #   'decoder.cnn_keys': 'image',
  #     'jax.platform': 'cpu',
  # })
  # if len(sys.argv) > 1:
  #   config['seed'] = int(sys.argv[1])

  config = embodied.Flags(config).parse()
  
  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(r'.*', logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  from pathlib import Path
  config.save(str(Path(logdir) / 'configs.yaml'))

  import crafter
  from gymnasium.wrappers.compatibility import EnvCompatibility
  from gym.wrappers.compatibility import EnvCompatibility
  from dreamerv3.embodied.envs import from_gym, from_gymnasium
#   env = "Pendulum-v1"  # Replace this with your Gym env.
# #   env = EnvCompatibility(env, render_mode='rgb_array') # Apply EnvCompatibility wrapper because crafter is still at gym==0.19.0 API
#   env = from_gym.FromGym(env, obs_key='vector', seed=0)  # Or obs_key='vector'.
  from dreamerv3.embodied.envs import simple_envs
  env = simple_envs.make('TwoRoomsFiftyFifty', seed=config.seed, goal_duration_episodes=600)
  # env = from_gymnasium.FromGymnasium('Taxi-v3', obs_key='vector')

  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.ParameterizedFifo(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length) # type: ignore
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
