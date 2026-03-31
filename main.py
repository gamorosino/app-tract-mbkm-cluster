#!/usr/bin/env python3
import json
import time
from pathlib import Path

from src.cluster80k import run_from_config
from src.io_utils import write_product_json


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_config()
    t0 = time.time()
    summary = run_from_config(cfg)
    summary["runtime_sec"] = round(time.time() - t0, 2)
    write_product_json(summary, "product.json")
    print("[main.py] done")


if __name__ == "__main__":
    main()
