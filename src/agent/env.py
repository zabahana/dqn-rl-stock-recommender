from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class MarketEnvConfig:
    price_data: Dict[str, pd.DataFrame]
    window: int = 20
    cash_fraction: float = 0.0


class MarketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: MarketEnvConfig) -> None:
        super().__init__()
        self.tickers = list(cfg.price_data.keys())
        self.window = cfg.window
        self.data = {t: df.copy() for t, df in cfg.price_data.items()}
        # Build aligned returns matrix
        self.aligned_returns = self._build_returns_matrix()
        num_assets = self.aligned_returns.shape[1]
        self.action_space = gym.spaces.Discrete(num_assets)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window * num_assets,), dtype=np.float32)
        self.t = 0
        self.position = None

    def _build_returns_matrix(self) -> np.ndarray:
        frames = []
        for t, df in self.data.items():
            col = "adj_close" if "adj_close" in df.columns else "close"
            s = df[col].pct_change().dropna().rename(t)
            frames.append(s)
        aligned = pd.concat(frames, axis=1).dropna()
        return aligned.to_numpy(dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.position = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        window_slice = self.aligned_returns[self.t - self.window : self.t]
        return window_slice.reshape(-1).astype(np.float32)

    def step(self, action: int):
        # Get basic return
        basic_return = float(self.aligned_returns[self.t, action])
        
        # Get sentiment score (placeholder - would need to be passed in or computed)
        sentiment_score = 0.0  # This would be computed from sentiment data
        
        # Get risk measure (volatility of the selected asset)
        asset_returns = self.aligned_returns[:, action]
        risk_measure = float(np.std(asset_returns))
        
        # Multi-factor reward: R_t = α·r_t + β·s_t - γ·σ_t
        alpha, beta, gamma = 1.0, 0.3, 0.2  # Weighting parameters
        reward = alpha * basic_return + beta * sentiment_score - gamma * risk_measure
        
        self.position = action
        self.t += 1
        terminated = self.t >= self.aligned_returns.shape[0]
        return self._obs(), reward, terminated, False, {}


