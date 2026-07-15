#!/usr/bin/env python3
"""
Show per-model summary weight for the last N epochs in a single table,
with start and end weights and each model's proportion of total.

Currently tracks: MiniMaxAI/MiniMax-M2.7 and moonshotai/Kimi-K2.6
"""

import json
import subprocess
import sys
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ARCHIVE_NODE = os.getenv("ARCHIVE_NODE_URL")
INFERENCED_BINARY = os.getenv("INFERENCED_BINARY")

if not ARCHIVE_NODE or not INFERENCED_BINARY:
    print("ERROR: ARCHIVE_NODE_URL and INFERENCED_BINARY must be set in .env", file=sys.stderr)
    sys.exit(1)

EPOCH_132_END_HEIGHT = 2058357
EPOCH_LENGTH = 15391

MODELS = [
    "MiniMaxAI/MiniMax-M2.7",
    "moonshotai/Kimi-K2.6",
]

SHORT = {
    "MiniMaxAI/MiniMax-M2.7": "Minimax",
    "moonshotai/Kimi-K2.6": "Kimi",
}


def epoch_start_height(epoch_id: int) -> int:
    return EPOCH_132_END_HEIGHT + (epoch_id - 133) * EPOCH_LENGTH + 1


def get_current_epoch() -> int:
    cmd = [INFERENCED_BINARY, "query", "inference", "get-current-epoch",
           "--node", ARCHIVE_NODE, "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return int(data.get("epoch") or data.get("current_epoch") or 0)
    cmd = [INFERENCED_BINARY, "status", "--node", ARCHIVE_NODE, "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    height = int(json.loads(result.stdout)["sync_info"]["latest_block_height"])
    return 132 + (height - EPOCH_132_END_HEIGHT) // EPOCH_LENGTH


def get_model_weights(poc_start_height: int, query_height: int) -> dict[str, int]:
    cmd = [
        INFERENCED_BINARY, "query", "inference",
        "all-ml-node-weight-distributions-for-stage",
        "--poc-stage-start-block-height", str(poc_start_height),
        "--height", str(query_height),
        "--node", ARCHIVE_NODE, "-o", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    by_model: dict[str, int] = {}
    for dist in data.get("distributions", []):
        mid = dist["model_id"]
        total = sum(w["weight"] for w in dist.get("weights", []))
        by_model[mid] = by_model.get(mid, 0) + total
    return by_model


def fmt_w(w: int) -> str:
    return f"{w:,}" if w else "-"


def fmt_pct(w: int, total: int) -> str:
    if not total:
        return "-"
    return f"{w / total * 100:.1f}%"


def main() -> None:
    epochs_back = int(sys.argv[1]) if len(sys.argv) > 1 else 14

    print("Fetching current epoch...", end=" ", flush=True)
    current_epoch = get_current_epoch()
    print(f"epoch {current_epoch}")

    first_epoch = current_epoch - epochs_back + 1
    print(f"Querying epochs {first_epoch}–{current_epoch} ({epochs_back} epochs)\n")

    # Collect data
    rows = []
    for epoch in range(first_epoch, current_epoch + 1):
        start_poc = epoch_start_height(epoch)
        end_poc = epoch_start_height(epoch + 1)

        print(f"  epoch {epoch}...", end=" ", flush=True)
        start_w = get_model_weights(start_poc, start_poc + 100)

        if epoch < current_epoch:
            end_w = get_model_weights(end_poc, end_poc + 100)
        else:
            end_w = None  # epoch in progress

        rows.append((epoch, start_w, end_w))
        print("ok" if start_w else "no data")

    # --- Print table ---
    # Columns: Epoch | Minimax start | Minimax start% | Kimi start | Kimi start% | Minimax end | ... | Kimi end | ...
    # Header groups: START OF EPOCH / END OF EPOCH

    print()

    # Column widths
    cE = 6    # epoch
    cW = 11   # weight
    cP = 7    # pct

    # Build header
    def col_headers():
        parts = [f"{'Epoch':>{cE}}"]
        for phase in ("START", "END"):
            for name in [SHORT[m] for m in MODELS]:
                parts.append(f"{name:>{cW}}")
                parts.append(f"{'%':>{cP}}")
        return "  ".join(parts)

    def phase_banner():
        # Phase labels above model columns
        blank = " " * cE
        start_span = (cW + 2 + cP + 2) * len(MODELS) - 2   # width of all START columns
        end_span = start_span
        return f"{blank}  {'── START ──':^{start_span}}  {'── END ──':^{end_span}}"

    header = col_headers()
    sep = "─" * len(header)

    print(phase_banner())
    print(header)
    print(sep)

    for epoch, start_w, end_w in rows:
        parts = [f"{epoch:>{cE}}"]

        for weights in (start_w, end_w):
            if weights is None:
                # epoch in progress
                for _ in MODELS:
                    parts.append(f"{'~':>{cW}}")
                    parts.append(f"{'~':>{cP}}")
            else:
                total = sum(weights.values())
                for m in MODELS:
                    w = weights.get(m, 0)
                    parts.append(f"{fmt_w(w):>{cW}}")
                    parts.append(f"{fmt_pct(w, total):>{cP}}")

        print("  ".join(parts))

    print(sep)
    print()


if __name__ == "__main__":
    main()
