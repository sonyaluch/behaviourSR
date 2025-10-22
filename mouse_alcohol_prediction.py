"""
This model analyzes historical alcohol consumption data per cycle per mouse and predicts future consumption patterns. Added some engineered features for better prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)


class MouseAlcoholPredictor:
  
    def __init__(self):
     
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.feature_names = None
        self.is_trained = False
    
    def generate_sample_data(self, n_mice=20, n_cycles=10): #generate sample data, number of mice and number of cycles, returns pd.DataFrame
       
        data = []
        
        for mouse_id in range(1, n_mice + 1): # mouse-based factors
            baseline_consumption = np.random.uniform(0.5, 5.0)
            tolerance_factor = np.random.uniform(0.9, 1.1)
            stress_sensitivity = np.random.uniform(0.5, 1.5)
            weight = np.random.uniform(20, 40)  # grams
            age = np.random.randint(8, 52)  # weeks
            
            for cycle in range(1, n_cycles + 1): # time-based factors
                cycle_effect = 1 + (cycle * 0.01 * tolerance_factor)
                
                # environmental factors
                stress_level = np.random.uniform(0, 1)
                temperature = np.random.uniform(20, 25)
                
                # Calculate consumption with realistic patterns
                consumption = (
                    baseline_consumption * cycle_effect +
                    stress_level * stress_sensitivity +
                    np.random.normal(0, 0.3) +  # random variation
                    (weight / 30) * 0.5  # weight factor
                )
                
                # ensure consumption not -ve
                consumption = max(0, consumption)
                
                data.append({
                    'mouse_id': mouse_id,
                    'cycle': cycle,
                    'weight': weight + np.random.normal(0, 0.5),  # slight weight variation
                    'age': age + (cycle // 7),  # mouse age increases over cycles
                    'stress_level': stress_level,
                    'temperature': temperature,
                    'previous_consumption': baseline_consumption if cycle == 1 else data[-1]['consumption'],
                    'consumption': consumption
                })
                
        return pd.DataFrame(data)
    
    def create_features(self, df):
       
        df = df.copy()
        
        # 3-cycle rolling consumption average by mouse, smooth out spikes
        df['consumption_ma_3'] = df.groupby('mouse_id')['consumption'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['consumption_ma_5'] = df.groupby('mouse_id')['consumption'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # tracks change in consumption by mouse
        df['consumption_trend'] = df.groupby('mouse_id')['consumption'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # cumulative consumption by mouse
        df['cumulative_consumption'] = df.groupby('mouse_id')['consumption'].transform('cumsum')
        
        # cycle squared to look for accelerating trends, log cycle to look for plateaus
        df['cycle_squared'] = df['cycle'] ** 2
        df['cycle_log'] = np.log1p(df['cycle'])
        
        # combined effects of weight and stress, age and cycle
        df['weight_stress'] = df['weight'] * df['stress_level']
        df['age_cycle'] = df['age'] * df['cycle']
        
        return df
    
    def prepare_data(self, df, target_col='consumption'): #prepare data for training
        
        # Features to use for prediction (excluding target and identifiers)
        feature_cols = [
            'cycle', 'weight', 'age', 'stress_level', 'temperature',
            'previous_consumption', 'consumption_ma_3', 'consumption_ma_5',
            'consumption_trend', 'cumulative_consumption', 'cycle_squared',
            'cycle_log', 'weight_stress', 'age_cycle'
        ]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, X_train, y_train, verbose=True):
       
        # scale features so all same range
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # train model, mark as trained
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        if verbose:
            # cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='r2'
            )
            print(f"Cross-validation R² scores: {cv_scores}")
            print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def predict(self, X):
       
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X) 
        return self.model.predict(X_scaled) # scale values because we trained on scaled
    
    def evaluate(self, X_test, y_test):
        
        predictions = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        return metrics, predictions
    
    def predict_future_cycles(self, mouse_data, n_future_cycles=5):
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        last_cycle_data = mouse_data.iloc[-1].copy()
        
        for i in range(1, n_future_cycles + 1):
            future_cycle = last_cycle_data['cycle'] + i
            future_data = pd.DataFrame([{
                'cycle': future_cycle,
                'weight': last_cycle_data['weight'],
                'age': last_cycle_data['age'] + (i // 7),
                'stress_level': np.random.uniform(0, 1),  # simulated stress
                'temperature': last_cycle_data['temperature'],
                'previous_consumption': last_cycle_data['consumption'],
                'consumption_ma_3': last_cycle_data['consumption_ma_3'],
                'consumption_ma_5': last_cycle_data['consumption_ma_5'],
                'consumption_trend': last_cycle_data['consumption_trend'],
                'cumulative_consumption': last_cycle_data['cumulative_consumption'],
                'cycle_squared': future_cycle ** 2,
                'cycle_log': np.log1p(future_cycle),
                'weight_stress': last_cycle_data['weight'] * last_cycle_data['stress_level'],
                'age_cycle': (last_cycle_data['age'] + (i // 7)) * future_cycle
            }])
            
            # predict
            predicted_consumption = self.predict(future_data[self.feature_names])[0]
            
            predictions.append({
                'cycle': future_cycle,
                'predicted_consumption': predicted_consumption
            })
            
            # update for next prediction
            last_cycle_data['consumption'] = predicted_consumption
            last_cycle_data['previous_consumption'] = predicted_consumption
            last_cycle_data['cumulative_consumption'] += predicted_consumption
        
        return pd.DataFrame(predictions)
    
    def get_feature_importance(self): # feature importance
     
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def visualize_results(data, predictions, mouse_id, future_predictions=None):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Mouse Alcohol Consumption Analysis', fontsize=16, fontweight='bold')
    
    # 1) actual vs model predicted (all data)
    axes[0, 0].scatter(data.loc[data.index, 'consumption'], predictions, alpha=0.5)
    axes[0, 0].plot([data['consumption'].min(), data['consumption'].max()],
                     [data['consumption'].min(), data['consumption'].max()],
                     'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Consumption (ml)')
    axes[0, 0].set_ylabel('Predicted Consumption (ml)')
    axes[0, 0].set_title('Actual vs Predicted Consumption')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2) time series for specific mouse
    mouse_data = data[data['mouse_id'] == mouse_id].copy()
    mouse_pred_idx = mouse_data.index
    mouse_predictions = predictions[data.index.isin(mouse_pred_idx)]
    
    axes[0, 1].plot(mouse_data['cycle'], mouse_data['consumption'], 
                    'o-', label='Actual', markersize=4)
    axes[0, 1].plot(mouse_data['cycle'], mouse_predictions, 
                    's-', label='Predicted', markersize=4, alpha=0.7)
    
    if future_predictions is not None:
        axes[0, 1].plot(future_predictions['cycle'], 
                        future_predictions['predicted_consumption'],
                        '^-', label='Future Predictions', markersize=6, color='red')
    
    axes[0, 1].set_xlabel('Cycle Number')
    axes[0, 1].set_ylabel('Consumption (ml)')
    axes[0, 1].set_title(f'Consumption Over Time - Mouse {mouse_id}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3) distribution of consumption across all mice
    axes[1, 0].hist(data['consumption'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Consumption (ml)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Alcohol Consumption')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4) average consumption by cycle (across all mice)
    avg_by_cycle = data.groupby('cycle')['consumption'].agg(['mean', 'std'])
    axes[1, 1].plot(avg_by_cycle.index, avg_by_cycle['mean'], 'o-', markersize=3)
    axes[1, 1].fill_between(avg_by_cycle.index, 
                             avg_by_cycle['mean'] - avg_by_cycle['std'],
                             avg_by_cycle['mean'] + avg_by_cycle['std'],
                             alpha=0.3)
    axes[1, 1].set_xlabel('Cycle Number')
    axes[1, 1].set_ylabel('Average Consumption (ml)')
    axes[1, 1].set_title('Average Consumption Across All Mice Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():  #orchestrator

    print("=" * 70)
    print("MOUSE ALCOHOL CONSUMPTION PREDICTION MODEL")
    print("=" * 70)
    print()
    
    # initialize predictor
    predictor = MouseAlcoholPredictor()
    
    # generate sample data
    print("Step 1: Generating sample data...")
    data = predictor.generate_sample_data(n_mice=20, n_cycles=10)
    print(f"Generated data for {data['mouse_id'].nunique()} mice over {data['cycle'].max()} cycles")
    print(f"Total samples: {len(data)}")
    print()
    
    # create features
    print("Step 2: Engineering features...")
    data = predictor.create_features(data)
    print(f"Created {len(predictor.feature_names) if predictor.feature_names else 14} features")
    print()
    
    # prepare data
    print("Step 3: Preparing data for training...")
    X, y = predictor.prepare_data(data)
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print()
    
    # train model
    print("Step 4: Training model...")
    print("-" * 70)
    predictor.train(X_train, y_train, verbose=True)
    print("-" * 70)
    print()
    
    # evaluate model
    print("Step 5: Evaluating model on test data...")
    metrics, test_predictions = predictor.evaluate(X_test, y_test)
    print(f"Mean Absolute Error: {metrics['mae']:.4f} ml")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f} ml")
    print(f"R² Score: {metrics['r2']:.4f}")
    print()
    
    # feature importance
    print("Step 6: Analyzing feature importance...")
    importance = predictor.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
    print()
    
    # predict future cycles by mouse
    print("Step 7: Predicting future consumption...")
    mouse_id = 5
    mouse_data = data[data['mouse_id'] == mouse_id]
    future_pred = predictor.predict_future_cycles(mouse_data, n_future_cycles=10)
    print(f"\nFuture predictions for Mouse {mouse_id}:")
    print(future_pred.to_string(index=False))
    print()
    
    # create visualizations
    print("Step 8: Creating visualizations...")
    test_data = data.loc[X_test.index]
    fig = visualize_results(test_data, test_predictions, mouse_id, future_pred)
    plt.savefig('/mnt/user-data/outputs/mouse_alcohol_predictions.png', 
                dpi=300, bbox_inches='tight')
    print("Visualization saved!")
    print()
    
    # save feature importance plot
    if importance is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.head(10).plot(x='feature', y='importance', kind='barh', ax=ax)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title('Top 10 Feature Importance Scores')
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/feature_importance.png', 
                    dpi=300, bbox_inches='tight')
        print("Feature importance plot saved!")
        print()
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return predictor, data


if __name__ == "__main__":  #only run the orchestrator if code is run directly
    predictor, data = main()
