import os
import gym
import numpy as np
#from roadenvacbeta import *
from gym import wrappers
from matplotlib import pyplot as plt
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_util import make_vec_env
# To check that your environment follows the gym interface, please use:
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
from loraEnv import loraEnv
from sb3_contrib import RecurrentPPO

def store(dir, rewards):
    # Create policy
    files = {dir: rewards}
    for filename in files.keys():
        file = open(filename, 'w')
        for i, k in enumerate(files[filename]):
            file.write(str(k) + '\n')
        file.close()

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, version, title='Learning Curve'):
    """
    Plot results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    # Save rewards in a file
    dir = 'rewards_v'+str(version)+'.txt'
    store(dir,y)

    fig = plt.figure(title)
    plt.plot(y, 'b')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward per episode')
    plt.title(title + " Smoothed / v:" + str(version))
    plt.savefig('results/reward.png')
    plt.show()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
        return True


log_dir = "logs/" # carpeta logs para meter todos los resultados
os.makedirs(log_dir, exist_ok=True)

# Don't forget that it has been load from RoadEnvAC, not RoadEnv
env = loraEnv(10)
"""TASK update env and input parameters"""

#### Validate the environment
# It will check your custom environment and output additional warnings if needed
check_env(env)

print("Observation space: ", env.observation_space)  # cual es el estado actual
print("Shape: ", env.observation_space.shape)  # la forma del estado
print("Action space: ", env.action_space)  # shape de la acción (número)

# The reset method is called at the beginning of an episode
obs = env.reset()  # se resetea. Tienes un nuevo estado, una nueva observación
# Sample a random action
action = env.action_space.sample()  # Coges una acción al azar
print("Sampled action: ", action)

obs, reward, done, info = env.step(action)

print(obs.shape, reward, done, info)  # pruebo que la recompensa sale bien

#### Training / Empieza el entrenamiento
env = wrappers.TimeLimit(env, max_episode_steps=10)  # Es lo que hace que cuando acabe el episodio resetee
env = Monitor(env, log_dir)
env = make_vec_env(lambda: env, n_envs=1)

callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, verbose=1)  # guarda el mejor modelo

#model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, gamma=0.9, learning_rate=0.0001)
#model = A2C('MlpPolicy', env, verbose=0, gamma=0.9, learning_rate=0.0001)
model = PPO('MlpPolicy', env, verbose=0, gamma=0.9, learning_rate=0.001, batch_size=512)
#model = SAC('MlpPolicy', env, verbose=0, gamma=0.9, learning_rate=0.0001, batch_size=128)

model.learn(total_timesteps=2000000, callback=callback)

version = 0

# Save the agent
model.save("lora_rl_sac_v" + str(version))

# Helper from the library
#results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO")
plot_results(log_dir, version)  # coge de la carpeta log el mejor modelo con la versión q le pongas y pinta la
#la recompensa q has tenido durante el tiempo de entrenamiento

# How to load previous trained model
#model = A2C.load("logs/best_model_v1.zip")
#model.set_env(env)