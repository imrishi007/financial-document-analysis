"""Data loading, target construction, preprocessing, and dataset pipelines."""

from src.data.document_dataset import DocumentChunkDataset, load_processed_filings
from src.data.multimodal_dataset import MultimodalAlignedDataset
from src.data.news_dataset import NewsWindowDataset, load_news_articles
from src.data.preprocessing import (
    FeatureScaler,
    SplitConfig,
    chunk_text,
    create_time_splits,
    fit_scaler,
    tokenize_short_text,
)
from src.data.price_dataset import (
    ENGINEERED_FEATURES,
    PriceWindowDataset,
    load_price_csv_dir,
    prepare_price_features,
)
from src.data.price_loader import PriceRequest, download_price_history
from src.data.target_builder import (
    build_binary_direction_labels,
    build_fundamental_surprise_targets,
    build_multi_horizon_direction_labels,
    build_realized_volatility_targets,
)

__all__ = [
    # Preprocessing
    "SplitConfig",
    "FeatureScaler",
    "create_time_splits",
    "fit_scaler",
    "chunk_text",
    "tokenize_short_text",
    # Price
    "PriceWindowDataset",
    "ENGINEERED_FEATURES",
    "load_price_csv_dir",
    "prepare_price_features",
    "PriceRequest",
    "download_price_history",
    # Documents
    "DocumentChunkDataset",
    "load_processed_filings",
    # News
    "NewsWindowDataset",
    "load_news_articles",
    # Multimodal
    "MultimodalAlignedDataset",
    # Targets
    "build_binary_direction_labels",
    "build_multi_horizon_direction_labels",
    "build_realized_volatility_targets",
    "build_fundamental_surprise_targets",
]
