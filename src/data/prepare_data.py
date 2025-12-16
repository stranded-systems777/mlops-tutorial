import sys
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from data_validation import DataValidator

# Add path manipulation after all imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Logging config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """main data preparation pipeline"""

    # initialize components
    loader = DataLoader("data/raw")
    validator = DataValidator()

    # load or generate data
    logger.info("step 1: loading data...")
    df = loader.load_sample_data()

    # validate schema
    logger.info("Step 2: Validating schema...")
    expected_columns = [
        "square_feet",
        "bedrooms",
        "bathrooms",
        "age",
        "location_score",
        "price",
    ]

    if not validator.validate_schema(df, expected_columns):
        raise ValueError("Schema validation failed")

    # validate data quality
    logger.info("Step 3: Validating data quality...")
    quality_results = validator.validate_data_quality(df)
    logger.info(f"Quality checks: {quality_results}")
    # save data
    logger.info("step 5: Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=12)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=12)

    # save splits (expand further for implementing it in azure)
    # Get the directory where prepare_data.py is located
    current_dir = Path(__file__).parent
    loader.data_path = (
        current_dir.parent / "data" / "processed"
    )  # ← This is always correct
    loader.save_data(train_df, "train.csv")
    loader.save_data(val_df, "val.csv")
    loader.save_data(test_df, "test.csv")

    logger.info("Data preparation complete")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
