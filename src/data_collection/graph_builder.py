from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data_collection.io_utils import ensure_directory


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_nodes_frame(companies: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": item["ticker"].upper(),
                "company_name": item["name"],
                "sector": "Technology",
            }
            for item in companies
        ]
    )


def build_default_edges() -> pd.DataFrame:
    # Directed edges with relation semantics and manually assigned prior weights.
    edges = [
        {"source": "NVDA", "target": "TSLA", "relation": "supplier_to", "weight": 0.95},
        {"source": "NVDA", "target": "MSFT", "relation": "supplier_to", "weight": 0.80},
        {"source": "NVDA", "target": "META", "relation": "supplier_to", "weight": 0.80},
        {"source": "NVDA", "target": "GOOGL", "relation": "supplier_to", "weight": 0.70},
        {"source": "INTC", "target": "MSFT", "relation": "supplier_to", "weight": 0.55},
        {"source": "AMD", "target": "MSFT", "relation": "supplier_to", "weight": 0.65},
        {"source": "AAPL", "target": "GOOGL", "relation": "platform_competitor", "weight": 0.90},
        {"source": "AAPL", "target": "MSFT", "relation": "platform_competitor", "weight": 0.80},
        {"source": "GOOGL", "target": "META", "relation": "ad_competitor", "weight": 0.90},
        {"source": "MSFT", "target": "GOOGL", "relation": "cloud_competitor", "weight": 0.85},
        {"source": "AMZN", "target": "MSFT", "relation": "cloud_competitor", "weight": 0.85},
        {"source": "AMZN", "target": "GOOGL", "relation": "cloud_competitor", "weight": 0.80},
        {"source": "TSLA", "target": "AAPL", "relation": "consumer_attention_overlap", "weight": 0.40},
        {"source": "ORCL", "target": "MSFT", "relation": "enterprise_competitor", "weight": 0.70},
        {"source": "ORCL", "target": "AMZN", "relation": "enterprise_competitor", "weight": 0.55},
        {"source": "MSFT", "target": "AAPL", "relation": "index_correlation", "weight": 0.60},
        {"source": "AAPL", "target": "MSFT", "relation": "index_correlation", "weight": 0.60},
        {"source": "GOOGL", "target": "AMZN", "relation": "index_correlation", "weight": 0.55},
        {"source": "AMZN", "target": "GOOGL", "relation": "index_correlation", "weight": 0.55},
    ]
    return pd.DataFrame(edges)


def build_graph_files(
    companies: list[dict[str, str]],
    nodes_csv: str | Path,
    edges_csv: str | Path,
) -> None:
    nodes_path = Path(nodes_csv)
    edges_path = Path(edges_csv)
    ensure_directory(nodes_path.parent)
    ensure_directory(edges_path.parent)

    nodes = build_nodes_frame(companies)
    edges = build_default_edges()

    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static graph node and edge files.")
    parser.add_argument("--config", type=str, default="configs/data_collection.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    build_graph_files(
        companies=config["companies"],
        nodes_csv=config["paths"]["graph_nodes_csv"],
        edges_csv=config["paths"]["graph_edges_csv"],
    )

    print(f"Graph nodes saved to {config['paths']['graph_nodes_csv']}")
    print(f"Graph edges saved to {config['paths']['graph_edges_csv']}")


if __name__ == "__main__":
    main()
