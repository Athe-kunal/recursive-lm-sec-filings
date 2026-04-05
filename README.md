# recursive-lm-sec-filings

Finance search RL environment now migrated from SkyRL to **Prime RL + Verifiers**.

## What changed

- SkyRL-specific environment wiring was replaced with a Verifiers-native custom multi-turn environment.
- `rlm_sec.envs.finance_env` now exposes `create_finance_env(...)` backed by `verifiers.MultiTurnEnv`.
- Tool execution is handled by async, HTTP-backed finance tools in `rlm_sec.envs.tools`.
- Prime-compatible environment loader is available at:
  - `rlm_sec.envs.prime_environment:load_environment`

## Prime RL setup

Follow Prime RL setup from the upstream README:

1. Clone Prime RL:
   ```bash
   git clone https://github.com/PrimeIntellect-ai/prime-rl.git
   cd prime-rl
   ```
2. Install dependencies:
   ```bash
   uv sync --all-extras
   ```
3. Install this environment package in editable mode:
   ```bash
   uv pip install -e /workspace/recursive-lm-sec-filings
   ```

## Running RL training with Prime RL

From the `prime-rl` repository:

```bash
bash /workspace/recursive-lm-sec-filings/run_train.sh
```

Set these optional variables before launching:

- `RLM_SEC_TRAIN_DATA` (default: `/workspace/recursive-lm-sec-filings/data/train.parquet`)
- `RLM_SEC_EVAL_DATA` (default: `/workspace/recursive-lm-sec-filings/data/validation.parquet`)
- `RLM_SEC_TOPK` (default: `3`)
- `RLM_SEC_TIMEOUT` (default: `30`)
- `RLM_SEC_MAX_QA_TURNS` (default: `4`)
- `RLM_SEC_MAX_RANKING_TURNS` (default: `1`)

## Environment protocol

The environment preserves the existing `<search>...</search>` and `<answer>...</answer>` protocol:

- `<search>SECFilingTool(query, ticker, year, filing_type)</search>`
- `<search>EarningsTranscriptTool(query, ticker, year, quarter)</search>`
- `<search>CompanyNameToTickerTool(company name)</search>`
- `<answer>final answer</answer>`

Rewards are still computed using the existing scorers in `rlm_sec.envs.rewards`.
