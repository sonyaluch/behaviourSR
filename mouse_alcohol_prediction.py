"""
Supervised Learning Model for Mouse Alcohol Consumption Prediction
This model analyzes historical alcohol consumption data per cycle per mouse
and predicts future consumption patterns.
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

# Set random seed for reproducibility
np.random.seed(42)


class MouseAlcoholPredictor:
    """
    A supervised learning model to predict mouse alcohol consumption.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor with specified model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use: 'random_forest', 'gradient_boost', or 'linear'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._initialize_model()
        self.feature_names = None
        self.is_trained = False
        
    def _initialize_model(self):
        """Initialize the machine learning model based on type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        elif self.model_type == 'gradient_boost':
            return GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate_sample_data(self, n_mice=20, n_cycles=50):
        """
        Generate synthetic mouse alcohol consumption data.
        
        Parameters:
        -----------
        n_mice : int
            Number of mice to simulate
        n_cycles : int
            Number of observation cycles
            
        Returns:
        --------
        pd.DataFrame
            Generated data with features and consumption amounts
        """
        data = []
        
        for mouse_id in range(1, n_mice + 1):
            # Each mouse has baseline characteristics
            baseline_consumption = np.random.uniform(0.5, 5.0)
            tolerance_factor = np.random.uniform(0.9, 1.1)
            stress_sensitivity = np.random.uniform(0.5, 1.5)
            weight = np.random.uniform(20, 40)  # grams
            age = np.random.randint(8, 52)  # weeks
            
            for cycle in range(1, n_cycles + 1):
                # Time-based factors
                cycle_effect = 1 + (cycle * 0.01 * tolerance_factor)
                
                # Environmental factors
                stress_level = np.random.uniform(0, 1)
                temperature = np.random.uniform(20, 25)
                
                # Calculate consumption with realistic patterns
                consumption = (
                    baseline_consumption * cycle_effect +
                    stress_level * stress_sensitivity +
                    np.random.normal(0, 0.3) +  # Random variation
                    (weight / 30) * 0.5  # Weight factor
                )
                
                # Ensure non-negative consumption
                consumption = max(0, consumption)
                
                data.append({
                    'mouse_id': mouse_id,
                    'cycle': cycle,
                    'weight': weight + np.random.normal(0, 0.5),  # Slight weight variation
                    'age': age + (cycle // 7),  # Age increases over time
                    'stress_level': stress_level,
                    'temperature': temperature,
                    'previous_consumption': baseline_consumption if cycle == 1 else data[-1]['consumption'],
                    'consumption': consumption
                })
                
        return pd.DataFrame(data)
    
    def create_features(self, df):
        """
        Create engineered features from raw data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw data with basic features
            
        Returns:
        --------
        pd.DataFrame
            Data with additional engineered features
        """
        df = df.copy()
        
        # Rolling averages (grouped by mouse)
        df['consumption_ma_3'] = df.groupby('mouse_id')['consumption'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['consumption_ma_5'] = df.groupby('mouse_id')['consumption'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # Consumption trend
        df['consumption_trend'] = df.groupby('mouse_id')['consumption'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Cumulative consumption
        df['cumulative_consumption'] = df.groupby('mouse_id')['consumption'].transform('cumsum')
        
        # Cycle-based features
        df['cycle_squared'] = df['cycle'] ** 2
        df['cycle_log'] = np.log1p(df['cycle'])
        
        # Interaction features
        df['weight_stress'] = df['weight'] * df['stress_level']
        df['age_cycle'] = df['age'] * df['cycle']
        
        return df
    
    def prepare_data(self, df, target_col='consumption'):
        """
        Prepare data for training by selecting features and target.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with all features
        target_col : str
            Name of target column
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
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
        """
        Train the model on provided data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training target
        verbose : bool
            Whether to print training information
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        if verbose:
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='r2'
            )
            print(f"Model: {self.model_type}")
            print(f"Cross-validation R² scores: {cv_scores}")
            print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Features to predict on
            
        Returns:
        --------
        np.array
            Predicted consumption values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.array
            Test features
        y_test : pd.Series or np.array
            True test values
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        return metrics, predictions
    
    def predict_future_cycles(self, mouse_data, n_future_cycles=10):
        """
        Predict alcohol consumption for future cycles for a specific mouse.
        
        Parameters:
        -----------
        mouse_data : pd.DataFrame
            Historical data for a specific mouse
        n_future_cycles : int
            Number of future cycles to predict
            
        Returns:
        --------
        pd.DataFrame
            Predictions for future cycles
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        last_cycle_data = mouse_data.iloc[-1].copy()
        
        for i in range(1, n_future_cycles + 1):
            # Create new cycle data
            future_cycle = last_cycle_data['cycle'] + i
            future_data = pd.DataFrame([{
                'cycle': future_cycle,
                'weight': last_cycle_data['weight'],
                'age': last_cycle_data['age'] + (i // 7),
                'stress_level': np.random.uniform(0, 1),  # Simulated stress
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
            
            # Predict
            predicted_consumption = self.predict(future_data[self.feature_names])[0]
            
            predictions.append({
                'cycle': future_cycle,
                'predicted_consumption': predicted_consumption
            })
            
            # Update for next prediction
            last_cycle_data['consumption'] = predicted_consumption
            last_cycle_data['previous_consumption'] = predicted_consumption
            last_cycle_data['cumulative_consumption'] += predicted_consumption
        
        return pd.DataFrame(predictions)
    
    def get_feature_importance(self):
        """
        Get feature importance (for tree-based models).
        
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def visualize_results(data, predictions, mouse_id, future_predictions=None):
    """
    Create visualizations of model results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full dataset
    predictions : np.array
        Model predictions
    mouse_id : int
        Specific mouse to visualize
    future_predictions : pd.DataFrame
        Future cycle predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Mouse Alcohol Consumption Analysis', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted (all data)
    axes[0, 0].scatter(data.loc[data.index, 'consumption'], predictions, alpha=0.5)
    axes[0, 0].plot([data['consumption'].min(), data['consumption'].max()],
                     [data['consumption'].min(), data['consumption'].max()],
                     'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Consumption (ml)')
    axes[0, 0].set_ylabel('Predicted Consumption (ml)')
    axes[0, 0].set_title('Actual vs Predicted Consumption')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series for specific mouse
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
    
    # 3. Distribution of consumption across all mice
    axes[1, 0].hist(data['consumption'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Consumption (ml)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Alcohol Consumption')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Average consumption by cycle (across all mice)
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


def main():
    """
    Main function to demonstrate the model.
    """
    print("=" * 70)
    print("MOUSE ALCOHOL CONSUMPTION PREDICTION MODEL")
    print("=" * 70)
    print()
    
    # Initialize predictor
    predictor = MouseAlcoholPredictor(model_type='random_forest')
    
    # Generate sample data
    print("Step 1: Generating sample data...")
    data = predictor.generate_sample_data(n_mice=20, n_cycles=50)
    print(f"Generated data for {data['mouse_id'].nunique()} mice over {data['cycle'].max()} cycles")
    print(f"Total samples: {len(data)}")
    print()
    
    # Create features
    print("Step 2: Engineering features...")
    data = predictor.create_features(data)
    print(f"Created {len(predictor.feature_names) if predictor.feature_names else 14} features")
    print()
    
    # Prepare data
    print("Step 3: Preparing data for training...")
    X, y = predictor.prepare_data(data)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print()
    
    # Train model
    print("Step 4: Training model...")
    print("-" * 70)
    predictor.train(X_train, y_train, verbose=True)
    print("-" * 70)
    print()
    
    # Evaluate model
    print("Step 5: Evaluating model on test data...")
    metrics, test_predictions = predictor.evaluate(X_test, y_test)
    print(f"Mean Absolute Error: {metrics['mae']:.4f} ml")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f} ml")
    print(f"R² Score: {metrics['r2']:.4f}")
    print()
    
    # Feature importance
    print("Step 6: Analyzing feature importance...")
    importance = predictor.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
    print()
    
    # Predict future cycles for a specific mouse
    print("Step 7: Predicting future consumption...")
    mouse_id = 5
    mouse_data = data[data['mouse_id'] == mouse_id]
    future_pred = predictor.predict_future_cycles(mouse_data, n_future_cycles=10)
    print(f"\nFuture predictions for Mouse {mouse_id}:")
    print(future_pred.to_string(index=False))
    print()
    
    # Create visualizations
    print("Step 8: Creating visualizations...")
    test_data = data.loc[X_test.index]
    fig = visualize_results(test_data, test_predictions, mouse_id, future_pred)
    plt.savefig('/mnt/user-data/outputs/mouse_alcohol_predictions.png', 
                dpi=300, bbox_inches='tight')
    print("Visualization saved!")
    print()
    
    # Save feature importance plot
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


if __name__ == "__main__":
    predictor, data = main()
