import pandas as pd
import numpy as np
from pathlib import Path
# from typing import Tuple
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and basic preprocessing"""

    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)

    def load_sample_data(self) -> pd.DataFrame:
        """Creates sample data for tutorial purposes
        In a real project, replace with actual data loading"""

        logger.info("Generate sample dataset...")

        np.random.seed(42)
        n_samples = 1000

        # create synthetic data (predicting housing prices)
        data = {
            "square_feet": np.random.normal(1500, 500, n_samples),
            "bedrooms": np.random.randint(1, 6, n_samples),
            "bathrooms": np.random.randint(1, 4, n_samples),
            "age": np.random.uniform(1, 10, n_samples),
            "location_score": np.random.uniform(0, 1, n_samples),
        }
        df = pd.DataFrame(data)

        # generate target variable
        df["price"] = (
            df["square_feet"] * 200
            + df["bedrooms"] * 10000
            + df["bathrooms"] * 15000
            - df["age"] * 500
            + df["location_score"] * 5000
            + np.random.normal(0, 10000, n_samples)
        )

        logger.info(f"Generated {len(df)} samples with {len(df.columns)} features")

        return df

    def save_data(self, df: pd.DataFrame, filename: str):
        """save df to csv"""
        # Create directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)

        filepath = self.data_path / filename
        df.to_csv(filepath, index=False)  # ← CORRECT! This saves/writes
        logger.info(f"Saved {len(df)} rows to {filepath}")
