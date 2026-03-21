"""SkyRL PPO entrypoint that uses SEC filings search as gym id ``search``.

SkyRL's built-in ``search`` environment calls a generic ``/retrieve`` API with a
single query. This repo's training data and ``server.py`` expect
``SECSearchEnv`` (query, ticker, year, filing_type) posting to
``/vector_store/search``. Importing ``skyrl_gym.envs`` and then replacing the
registry entry keeps ``environment.env_class=search`` and
``environment.skyrl_gym.search.*`` Hydra keys unchanged.
"""

import skyrl_gym.envs  # noqa: F401 — load default gym registrations
from skyrl_gym.envs.registration import EnvSpec, registry

registry["search"] = EnvSpec(
    id="search",
    entry_point="rlm_sec.envs.sec_filings_env:SECSearchEnv",
)

from skyrl.train.entrypoints.main_base import main

if __name__ == "__main__":
    main()
