# AGENTS.md

## Cursor Cloud specific instructions

AlphaGPT is a Python crypto-quant system (see `CATREADME.md` for the module map).
Modules: `data_pipeline` (Birdeye→Postgres), `model_core` (Transformer factor
mining — the core), `strategy_manager` + `execution` (live Solana trading),
`dashboard` (Streamlit GUI). There are no automated tests, lint config, or CI in
this repo. All modules use package-relative imports, so run them as modules with
`python3 -m <pkg>.<mod>` from the repo root.

### What can and cannot run in this environment
- Runnable end-to-end here: `model_core` factor mining and the Streamlit
  `dashboard` (both only need Postgres + seeded data).
- Not runnable here: `data_pipeline.run_pipeline` needs a paid `BIRDEYE_API_KEY`;
  `strategy_manager.runner` / `execution.trader` need a funded Solana wallet
  (`SOLANA_PRIVATE_KEY`) + RPC and place real trades. These modules import fine
  but cannot be exercised without those external credentials/funds.

### Dependency caveats (important, non-obvious)
- The update script installs everything; you normally do not reinstall. Key pins
  it enforces on top of `requirements.txt` (which is under-constrained):
  - `torch` is the CPU build (installed from the PyTorch CPU index); there is no GPU.
  - `pandas` must be `<2.3`. The code uses `DataFrame.fillna(method='ffill')`
    (`model_core/data_loader.py`), which pandas 3.0 removed. `requirements.txt`
    only says `pandas>=2.0.0`, so an unpinned install breaks the data loader.
  - `solana==0.30.2` / `solders==0.18.1`. The trading code imports
    `TokenAccountOpts` from `solana.rpc.types`; solana-py ≥0.40 moved it to
    `solana.rpc.models`, which breaks `execution.trader` and
    `strategy_manager.runner`. These pins keep the whole codebase importable.
- `~/.local/bin` is NOT on `PATH`. Run console entry points as modules, e.g.
  `python3 -m streamlit ...` (not bare `streamlit`).

### PostgreSQL (system dependency, required)
- Postgres is a system package (not in the update script). If a fresh VM lacks it:
  `sudo apt-get install -y postgresql postgresql-contrib`.
- Start it (no systemd in the container): `sudo pg_ctlcluster 16 main start`.
- App defaults (in every module's config, overridable via env/`.env`):
  user `postgres`, password `password`, host `localhost`, port `5432`, db
  `crypto_quant`. Set the password + create the DB once:
  `sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'password';"` and
  `sudo -u postgres createdb crypto_quant`.
- The schema (`tokens`, `ohlcv`) is auto-created by `DBManager.init_schema()`.
  TimescaleDB is not installed; the code logs a warning and falls back to plain
  Postgres — this is expected, not an error.

### Seeding data for local runs
- The DB starts empty; `model_core` and the dashboard need rows. Seed synthetic
  data by connecting with `DBManager`, calling `init_schema()`, `upsert_tokens()`
  and `batch_insert_ohlcv()`.
- Gotcha: `CryptoDataLoader` pivots all tokens onto a shared time index and
  forward-fills, then `fillna(0.0)`. If seeded tokens do not share the same
  timestamps, missing cells become `0.0` open prices and `target_ret =
  log(...)` produces `-inf`/`nan`, which makes training rewards `nan`. Seed every
  token on an identical timestamp grid. Also keep `liquidity > 5e5`
  (`model_core/backtest.py` `min_liq`) or the backtest takes no positions.

### Running the services
- Factor-mining engine (core): `python3 -m model_core.engine`. Defaults are
  GPU-scale (`ModelConfig.BATCH_SIZE=8192`, `TRAIN_STEPS=1000`) and infeasible on
  CPU; for a quick CPU run, set `ModelConfig.BATCH_SIZE`/`TRAIN_STEPS` small at
  runtime before calling `AlphaEngine().train()`. Outputs `best_meme_strategy.json`
  and `training_history.json` in the working directory.
- Dashboard (Streamlit GUI): from the repo root run
  `PYTHONPATH=dashboard python3 -m streamlit run dashboard/app.py --server.port 8501`.
  `dashboard/app.py` uses top-level imports (`data_service`, `visualizer`), so
  `dashboard/` must be on `PYTHONPATH`; run from the repo root so it can read
  `best_meme_strategy.json` from the cwd. Health check: `/_stcore/health`.
