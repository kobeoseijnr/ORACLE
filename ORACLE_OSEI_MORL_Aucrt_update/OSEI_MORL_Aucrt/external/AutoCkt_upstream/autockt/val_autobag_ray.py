import ray
import ray.tune as tune
from ray.rllib.algorithms.ppo import PPOConfig

import os
import sys

# Ensure repo root is on PYTHONPATH so `import autockt...` works
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args = parser.parse_args()
ray.init(runtime_env={"working_dir": _REPO_ROOT})

#configures training of the agent with associated hyperparameters
#See Ray documentation for details on each parameter
config_train = {
            #"sample_batch_size": 200,
            "train_batch_size": 1200,
            #"sgd_minibatch_size": 1200,
            #"num_sgd_iter": 3,
            #"lr":1e-3,
            #"vf_loss_coeff": 0.5,
            "horizon":  30,
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 6,
            "env_config":{"generalize":True, "run_valid":False},
            }

#Runs training and saves the result in ~/ray_results/train_ngspice_45nm
#If checkpoint fails for any reason, training can be restored 
if not args.checkpoint_dir:
    algo_config = (
        PPOConfig()
        .environment(env=TwoStageAmp, env_config=config_train.get("env_config", {}))
        .framework("torch")
        .env_runners(num_env_runners=int(config_train.get("num_workers", 0)))
        .resources(num_gpus=int(config_train.get("num_gpus", 0)))
        .training(
            train_batch_size=int(config_train.get("train_batch_size", 1200)),
            model=config_train.get("model", {"fcnet_hiddens": [64, 64]}),
        )
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=tune.RunConfig(
            name="train_45nm_ngspice",
            stop={"episode_reward_mean": -0.02},
            checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=1),
        ),
    )
    results = tuner.fit()
else:
    print("RESTORING NOW!!!!!!")
    algo_config = (
        PPOConfig()
        .environment(env=TwoStageAmp, env_config=config_train.get("env_config", {}))
        .framework("torch")
        .env_runners(num_env_runners=int(config_train.get("num_workers", 0)))
        .resources(num_gpus=int(config_train.get("num_gpus", 0)))
        .training(
            train_batch_size=int(config_train.get("train_batch_size", 1200)),
            model=config_train.get("model", {"fcnet_hiddens": [64, 64]}),
        )
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=tune.RunConfig(
            name="restore_ppo",
            stop={"episode_reward_mean": -0.02},
            checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=1),
        ),
        restore=args.checkpoint_dir,
    )
    results = tuner.fit()
