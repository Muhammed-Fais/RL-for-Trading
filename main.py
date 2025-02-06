import ray
from ray.rllib.algorithms.ppo import PPOConfig
from utils.data_loader import get_price_data
from envs.trading_env import CryptoTradingEnv
from ray.tune.registry import register_env

# Initialize Ray
ray.init()

# Load data from Yahoo Finance
data = get_price_data("BTC-USD", start_date="2022-01-01", end_date="2023-01-01")

# Environment registration
def create_env(config):
    return CryptoTradingEnv(config)

register_env("CryptoTradingEnv", create_env)

# Configuration for PPO
config = (
    PPOConfig()
    .environment(env="CryptoTradingEnv", env_config={
        "data": data,
        "initial_balance": 1000
    })
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .training(
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu"
        },
        train_batch_size=4000
    )
)

# Train the PPO agent
algo = config.build()
for i in range(10):
    result = algo.train()
    print(f"Iteration {i}: reward_mean = {result['episode_reward_mean']}")

# Save the trained model
checkpoint = algo.save("./ppo_trading_model")
print(f"Checkpoint saved at {checkpoint}")

algo.stop()
ray.shutdown()