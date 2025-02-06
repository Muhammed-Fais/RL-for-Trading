import numpy as np
import gym
from ray.rllib.env.env_context import EnvContext

class CryptoTradingEnv(gym.Env):
    """
    Custom RL environment for crypto trading.
    """
    def __init__(self, config: EnvContext):
        super().__init__()
        self.data = config.get("data", np.random.random((1000, 3)))
        self.initial_balance = config.get("initial_balance", 1000)
        self.current_step = 0
        self.done = False
        self.current_balance = self.initial_balance
        self.position = None

        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.done = False
        self.current_balance = self.initial_balance
        self.position = None
        return self._get_observation()

    def _get_observation(self):
        return np.array([
            self.data[self.current_step][0],
            self.current_balance,
            1 if self.position == 'long' else -1 if self.position == 'short' else 0
        ])

    def step(self, action):
        price = self.data[self.current_step][0]
        reward = 0
        
        if action == 0:
            if self.position is None:
                self.position = 'long'
                self.entry_price = price

        elif action == 1:
            if self.position == 'long':
                reward = price - self.entry_price
                self.current_balance += reward
                self.position = None
            elif self.position is None:
                self.position = 'short'
                self.entry_price = price

        elif action == 2:
            reward = 0

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}