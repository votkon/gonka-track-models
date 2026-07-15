# gonka-track-models

CLI tool to track per-model summary weights across epochs on the Gonka inference network.

Currently tracks:
- `MiniMaxAI/MiniMax-M2.7`
- `moonshotai/Kimi-K2.6`

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

**.env variables:**

| Variable | Description |
|---|---|
| `ARCHIVE_NODE_URL` | RPC endpoint of an archive node (e.g. `http://host:26657`) |
| `INFERENCED_BINARY` | Path to the `inferenced` CLI binary |

## Usage

```bash
python model_weights.py [epochs_back]
```

`epochs_back` defaults to `14`. Example:

```
python model_weights.py 10
```

Prints a table showing total network weight and each model's share per epoch.

- **Total** — root `epoch_group_data.total_weight`, matching the value shown on the tracker.
- **Per-model split** — each model's share of the total, derived from `confirmation_weight` ratios in the model subgroups (`confirmation_weight` = raw PoC weight scaled by the per-model `WeightScaleFactor`).
