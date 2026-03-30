"""Legacy compatibility module.

SkyRL is no longer used in this repository. Use Prime RL with:
`rlm_sec.envs.prime_environment:load_environment`.
"""

from rlm_sec.envs.prime_environment import load_environment

__all__ = ["load_environment"]
