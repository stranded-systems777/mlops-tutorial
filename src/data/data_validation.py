import pandas as pd
import great_expectations as ge
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and schema"""

    def __init__(self):
        self.expectations = []

    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """check if df has expected columns"""

        missing_cols = set(expected_columns) - set(df.columns)

        if missing_cols:
            logger.error(f"Missign columns: {missing_cols}")
            return False

        logger.info("Schema validation passed")
        return True

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Run data quality checks"""
        ge_df = ge.from_pandas(df)
        results = {}

        # check for nulls
        for col in df.columns:
            result = ge_df.expect_column_values_to_not_be_null(col)
            results[f"{col}_not_null"] = result.success

        # check value ranges for numeric columns
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            if col != "price":  # skip target variable
                result = ge_df.expect_column_values_to_be_between(
                    col, min_value=df[col].min(), max_value=df[col].max()
                )
                results[f"{col}_in_range"] = result.success

            all_passed = all(results.values())
            status = "PASSED" if all_passed else "FAILED"
            logger.info(f"Data quality validation: {status}")
            return results
