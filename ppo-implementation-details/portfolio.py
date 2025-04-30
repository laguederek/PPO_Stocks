import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, price_data):
        super(PortfolioEnv, self).__init__()
        self.price_data = price_data
        self.n_assets = price_data.shape[1]
        self.max_steps = len(price_data) - 1

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32)

        self.current_step = 0
        self.initial_value = 1.0
        self.portfolio_value = self.initial_value

    def reset(self, seed=None, return_info=False, options=None):
        self.current_step = 1  # Start at step 1 so we can look back at t-1
        self.portfolio_value = self.initial_value
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = self.price_data.iloc[self.current_step].values  # row as numpy array
        obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        action = action / (np.sum(action) + 1e-6)

        # Compute price relatives (current price / previous price)
        prev_prices = self.price_data.iloc[self.current_step - 1]
        current_prices = self.price_data.iloc[self.current_step]
        price_relatives = current_prices / prev_prices

        # Portfolio return for this step
        portfolio_return = np.dot(action, price_relatives)
        self.portfolio_value *= portfolio_return

        # Reward = log return (common for numerical stability)
        epsilon = 1e-8
        reward = np.log(np.maximum(portfolio_return, epsilon))

        self.current_step += 1
        done = self.current_step >= self.max_steps
        info = {}

        if done:
            total_return = (self.portfolio_value - self.initial_value) / self.initial_value

            info["total_return"] = total_return
            info["steps_taken"] = self.current_step

        return self._get_obs(), reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.4f}")
