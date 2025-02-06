# Crypto Trading RL with PPO and RLlib
This repository demonstrates reinforcement learning using RLlib for cryptocurrency trading.

## Structure
- `main.py`: Main script for training the RL agent.
- `envs/trading_env.py`: Custom trading environment.
- `utils/data_loader.py`: Helper functions to fetch market data from Yahoo Finance.
- `ppo_trading_model`: Directory to store saved PPO models.

## Installation
```bash
pip install ray[rllib] yfinance torch numpy gym
```

## Running
```bash
python main.py
```

## Future Improvements
- Include advanced reward functions.
- Integrate live data for real-world trading.
