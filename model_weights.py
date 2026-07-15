#!/usr/bin/env python3
"""
Show per-model weight distribution for the last N epochs.

Total network weight comes from the root epoch_group_data.total_weight — the
same value the tracker shows. Per-model split uses confirmation_weight ratios
from each model's subgroup.

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

MODELS = [
    "MiniMaxAI/MiniMax-M2.7",
    "moonshotai/Kimi-K2.6",
]

SHORT = {
    "MiniMaxAI/MiniMax-M2.7": "Minimax",
    "moonshotai/Kimi-K2.6": "Kimi",
}


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
    EPOCH_132_END_HEIGHT = 2058357
    EPOCH_LENGTH = 15391
    return 132 + (height - EPOCH_132_END_HEIGHT) // EPOCH_LENGTH


def query_egdata(epoch_index: int, model_id: str | None = None) -> dict | None:
    cmd = [
        INFERENCED_BINARY, "query", "inference",
        "show-epoch-group-data", str(epoch_index),
        "--node", ARCHIVE_NODE, "-o", "json",
    ]
    if model_id:
        cmd += ["--model-id", model_id]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data.get("epoch_group_data") or data.get("EpochGroupData") or data


def get_epoch_data(epoch_index: int) -> dict | None:
    """
    Returns {model_id: allocated_weight} where allocated_weight is each model's
    share of root total_weight, split by confirmation_weight ratios from subgroups.
    Also returns '_total' key with the root total_weight.
    """
    root = query_egdata(epoch_index)
    if root is None:
        return None
    total_weight = int(root.get("total_weight") or 0)
    if not total_weight:
        return None

    cw: dict[str, int] = {}
    for model_id in MODELS:
        sub = query_egdata(epoch_index, model_id)
        if sub is None:
            continue
        weights = sub.get("validation_weights") or []
        cw[model_id] = sum(int(w.get("confirmation_weight", 0)) for w in weights)

    cw_total = sum(cw.values())
    if not cw_total:
        return None

    result: dict[str, int] = {"_total": total_weight}
    for model_id in MODELS:
        result[model_id] = round(cw.get(model_id, 0) / cw_total * total_weight)
    return result


def fmt_w(w: int | None) -> str:
    if w is None:
        return "-"
    return f"{w:,}"


def fmt_pct(w: int | None, total: int) -> str:
    if w is None or not total:
        return "-"
    return f"{w / total * 100:.1f}%"


def main() -> None:
    epochs_back = int(sys.argv[1]) if len(sys.argv) > 1 else 14

    print("Fetching current epoch...", end=" ", flush=True)
    current_epoch = get_current_epoch()
    print(f"epoch {current_epoch}")

    first_epoch = current_epoch - epochs_back + 1
    print(f"Querying epochs {first_epoch}–{current_epoch} ({epochs_back} epochs)\n")

    rows = []
    for epoch in range(first_epoch, current_epoch + 1):
        print(f"  epoch {epoch}...", end=" ", flush=True)
        data = get_epoch_data(epoch)
        rows.append((epoch, data))
        print("ok" if data else "no data")

    print()

    cE = 6
    cW = 11
    cP = 7
    cT = 12  # total weight column

    header = "  ".join([
        f"{'Epoch':>{cE}}",
        f"{'Total':>{cT}}",
        *[part for m in MODELS for part in (f"{SHORT[m]:>{cW}}", f"{'%':>{cP}}")],
    ])
    sep = "─" * len(header)

    print(header)
    print(sep)

    for epoch, data in rows:
        if data is None:
            parts = [f"{epoch:>{cE}}", f"{'-':>{cT}}"]
            for _ in MODELS:
                parts += [f"{'-':>{cW}}", f"{'-':>{cP}}"]
        else:
            total = data["_total"]
            parts = [f"{epoch:>{cE}}", f"{fmt_w(total):>{cT}}"]
            for m in MODELS:
                w = data.get(m)
                parts.append(f"{fmt_w(w):>{cW}}")
                parts.append(f"{fmt_pct(w, total):>{cP}}")
        print("  ".join(parts))

    print(sep)
    print()


if __name__ == "__main__":
    main()
