import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handle model training with MLflow tracking"""
    def __init__(self, experiment_name: str="house-price-prediction"):
        mlflow.set_experiment(experiment_name)
        self.scaler = StandardScaler()
        
    def load_data(self, split:str='train')->tuple:
        """load processed data"""
        df = pd.read_csv(f'data/processed/{split}.csv')
        X = df.drop('price', axis=1)
        y = df['price']
        return X,y
    
    def train_model(self, model_type:str='random_forest', params: dict = None):
        """train model with mlflow"""

        with mlflow.start_run(run_name=f"{model_type}_training"):
            # load data 
            logger.info("Loading training data...")
            X_train,y_train = self.load_data('train')
            X_val, y_val = self.load_data('val')

            #scale features
            logger.info('Scaling features...')
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # load data info
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param('n_features', X_train.shape[1])
            mlflow.log_param('model_type', model_type)

            # initialize model 
            if model_type == 'random_forest':
                params=params or {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 12
                }
                model = RandomForestRegressor(**params)
            else: 
                params = params or {}
                model = LinearRegression(**params)
            
            # log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)

            # train model
            logger.info(f"Training {model_type} model...")
            model.fit(X_train_scaled, y_train)

            # evaluate on training set
            train_preds = model.predict(X_train_scaled)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            train_r2 = r2_score(y_train, train_preds)
            train_mae = mean_absolute_error(y_train,train_preds)

            # evaluate on validation set 
            val_preds = model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val,val_preds))
            val_r2 = r2_score(y_val, val_preds)
            val_mae = mean_absolute_error(y_val, val_preds)

            # log metrics 
            mlflow.log_metric('train_rmse', train_rmse)
            mlflow.log_metric('train_r2', train_r2)
            mlflow.log_metric('train_mae', train_mae)
            mlflow.log_metric('val_rmse', val_rmse)
            mlflow.log_metric('val_r2', val_r2)
            mlflow.log_metric('val_mae', val_mae)

            logger.info(f"validation rmse: {val_rmse}")
            logger.info(f'val r2: {val_r2}')

            # save model and scaler
            joblib.dump(model, "models/model.pkl")
            joblib.dump(self.scaler, 'models/scaler.pkl')

            # log model 
            mlflow.sklearn.log_model(model, 'model')

            # log feature importance (for random forest)
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(X_train.columns,
                                           model.feature_importances_))
                with open('models/feature_importance.json', 'w') as f: 
                    json.dump(importance_dict, f , indent=2)
                mlflow.log_artifact('models/feature_importance.json')
            
            return model, self.scaler
        
def main():
    """train multiple models and compare"""

    trainer = ModelTrainer()

    # train random forest 
    logger.info('\n' + '='*50)
    logger.info("training Random Forest Model")
    logger.info('\n' + '='*50)

    rf_params = {
        'n_estimators':100,
        'max_depth': 10,
        'min_samples_split': 5, 
        'random_state': 12
    }
    trainer.train_model('random_forest', rf_params)

    # # train linear regression
    # logger.info('\n' + '='*50)
    # logger.info("training Linear Regression Model")
    # logger.info('\n' + '='*50)

    # trainer.scaler = StandardScaler() # reset the scaler
    # trainer.train_model('linear_regression')

    # logger.info('\n' + '='*50)
    # logger.info('training complete, view results with mlflow ui')
    # logger.info('\n' + '='*50)

if __name__ == "__main__":
    main()

                