import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import joblib 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """evaluate trained models"""
    def __init__(self, model_path: str= 'models/model.pkl', scaler_path: str='models/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def evaluate_on_test_set(self):
        """final evaluation on test set"""

        # load test data
        logger.info('Loading test data...')
        test_df = pd.read_csv('data/processed/test.csv')
        X_test = test_df.drop('price', axis=1)
        y_test = test_df['price']

        # scale features
        X_test_scaled = self.scaler.transform(X_test)

        # make predictions
        test_preds = self.model.predict(X_test_scaled)
    

        # calculate metrics
        r2 = r2_score(y_test, test_preds)
        mae = mean_absolute_error(y_test, test_preds)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        mape = np.mean(np.abs((y_test-test_preds)/y_test)*100)

        logger.info("\n" + "="*50)
        logger.info("TEST SET RESULTS")
        logger.info("="*50)
        logger.info(f"RMSE: ${rmse:,.2f}")
        logger.info(f"MAE: ${mae:,.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info("="*50)


        # create visualization 
        self._plot_predictions(y_test, test_preds)
 
        return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
        }
        
    def _plot_predictions(self, y_test, test_preds):
        """Create prediction plots"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Predicted vs Actual
        axes[0].scatter(y_test, test_preds, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', lw=2)
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title('Predicted vs Actual price')
        
        # Residuals
        residuals = y_test - test_preds
        axes[1].scatter(test_preds, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Price')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig('models/evaluation_plots.png', dpi=300)
        logger.info("Evaluation plots saved to models/evaluation_plots.png")

def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate_on_test_set()

if __name__ == "__main__":
    main()