"""
Gymnasium compatibility shim.

The environment follows the Gymnasium API exactly -- ``reset(seed=..., options=...)
-> (obs, info)`` and ``step(action) -> (obs, reward, terminated, truncated, info)``
-- but does **not** require Gymnasium to be installed. If it is present we use
the real ``gymnasium.spaces`` and subclass ``gymnasium.Env``, so the env drops
straight into Stable-Baselines3 / RLlib / CleanRL. If it is absent we fall back
to minimal duck-typed stand-ins with the same attributes and ``sample()`` /
``contains()`` methods, and the env works with numpy alone.

Check :data:`GYMNASIUM_AVAILABLE` if you need to branch.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - depends on the user's environment
    from gymnasium import Env as GymEnv
    from gymnasium.spaces import Box, Discrete

    GYMNASIUM_AVAILABLE = True

except ImportError:  # pragma: no cover - exercised when gymnasium is absent

    GYMNASIUM_AVAILABLE = False

    class GymEnv:  # type: ignore[no-redef]
        """Stand-in base class used when Gymnasium is not installed.

        Declares the attributes Gymnasium's ``Env`` would provide so that
        downstream code (and type checkers) see a consistent surface.
        """

        metadata: dict = {"render_modes": []}
        render_mode: Optional[str] = None
        action_space: Any = None
        observation_space: Any = None

        def reset(self, *, seed=None, options=None):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def render(self):
            raise NotImplementedError

        def close(self) -> None:
            return None

    class Discrete:  # type: ignore[no-redef]
        """Minimal replacement for ``gymnasium.spaces.Discrete``."""

        def __init__(self, n: int, seed: Optional[int] = None) -> None:
            self.n = int(n)
            self.shape: Tuple[int, ...] = ()
            self.dtype = np.int64
            self._rng = np.random.default_rng(seed)

        def sample(self) -> int:
            """Uniformly sample a valid action index."""
            return int(self._rng.integers(0, self.n))

        def contains(self, x: Any) -> bool:
            """True if ``x`` is a valid index in ``[0, n)``."""
            try:
                xi = int(x)
            except (TypeError, ValueError):
                return False
            return 0 <= xi < self.n

        def seed(self, seed: Optional[int] = None) -> None:
            """Reseed the sampler."""
            self._rng = np.random.default_rng(seed)

        def __contains__(self, x: Any) -> bool:
            return self.contains(x)

        def __repr__(self) -> str:
            return f"Discrete({self.n})"

    class Box:  # type: ignore[no-redef]
        """Minimal replacement for ``gymnasium.spaces.Box``."""

        def __init__(
            self,
            low,
            high,
            shape: Optional[Sequence[int]] = None,
            dtype=np.float32,
            seed: Optional[int] = None,
        ) -> None:
            self.dtype = np.dtype(dtype)
            if shape is None:
                low_arr = np.asarray(low, dtype=self.dtype)
                shape = low_arr.shape
            self.shape = tuple(int(s) for s in shape)
            self.low = np.broadcast_to(
                np.asarray(low, dtype=self.dtype), self.shape
            ).copy()
            self.high = np.broadcast_to(
                np.asarray(high, dtype=self.dtype), self.shape
            ).copy()
            self._rng = np.random.default_rng(seed)

        def sample(self) -> np.ndarray:
            """Uniformly sample a point in the box."""
            return self._rng.uniform(
                low=self.low, high=self.high, size=self.shape
            ).astype(self.dtype)

        def contains(self, x: Any) -> bool:
            """True if ``x`` has the right shape and lies within bounds."""
            arr = np.asarray(x)
            return (
                arr.shape == self.shape
                and bool(np.all(arr >= self.low - 1e-6))
                and bool(np.all(arr <= self.high + 1e-6))
            )

        def seed(self, seed: Optional[int] = None) -> None:
            """Reseed the sampler."""
            self._rng = np.random.default_rng(seed)

        def __contains__(self, x: Any) -> bool:
            return self.contains(x)

        def __repr__(self) -> str:
            return f"Box({self.shape}, {self.dtype})"


__all__ = ["Box", "Discrete", "GymEnv", "GYMNASIUM_AVAILABLE"]
