#!/usr/bin/env python3
import json
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_product_json(summary, path="product.json"):
    cluster_stats = summary.get("cluster_stats", {})
    payload = {
        "brainlife": [
            {
                "type": "info",
                "msg": f"Completed clustering for {summary.get('input_streamlines', 0)} streamlines into {summary.get('n_clusters_final', 0)} clusters."
            }
        ],
        "meta": {
            "fold_id": summary.get("fold_id"),
            "distance": summary.get("distance"),
            "n_clusters_requested": summary.get("n_clusters_requested"),
            "n_clusters_final": summary.get("n_clusters_final"),
            "input_streamlines": summary.get("input_streamlines"),
            "runtime_sec": summary.get("runtime_sec"),
            "singleton_clusters": cluster_stats.get("singleton_clusters"),
            "cluster_size_min": cluster_stats.get("min"),
            "cluster_size_median": cluster_stats.get("median"),
            "cluster_size_max": cluster_stats.get("max")
        }
    }
    write_json(path, payload)
