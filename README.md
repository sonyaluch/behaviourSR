# Mouse Alcohol Consumption Prediction Model

This machine learning model analyzes historical alcohol consumption data for laboratory mice, and then uses it to predict future consumption patterns based on physiological and environmental factors.

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn --break-system-packages
```

Or install via requirements file:

```bash
pip install -r requirements.txt --break-system-packages
```

### Required Packages

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Usage

### Basic Usage

Run the complete analysis pipeline:

```bash
python mouse_alcohol_prediction__2_.py
```

This will:
1. Generate sample data for 20 mice over 10 cycles
2. Engineer features
3. Train the Random Forest model
4. Evaluate performance on test data
5. Generate predictions for future cycles
6. Create and save visualizations

### Using in Jupyter Notebook

```python
from mouse_alcohol_prediction import MouseAlcoholPredictor

# Initialize predictor
predictor = MouseAlcoholPredictor()

# Generate sample data
data = predictor.generate_sample_data(n_mice=20, n_cycles=10)

# Create engineered features
data = predictor.create_features(data)

# Prepare data for training
X, y = predictor.prepare_data(data)

# Split into train/test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
predictor.train(X_train, y_train)

# Evaluate performance
metrics, predictions = predictor.evaluate(X_test, y_test)
print(f"R² Score: {metrics['r2']:.4f}")

# Predict future cycles for a specific mouse
mouse_data = data[data['mouse_id'] == 5]
future_predictions = predictor.predict_future_cycles(mouse_data, n_future_cycles=10)
print(future_predictions)
```

## Data Structure

### Input Features

The model uses the following features for prediction:

**Primary Features:**
- `cycle`: Experimental cycle number
- `weight`: Mouse weight in grams
- `age`: Mouse age in weeks
- `stress_level`: Measured stress level (0-1)
- `temperature`: Environmental temperature (°C)
- `previous_consumption`: Consumption from previous cycle

**Engineered Features:**
- `consumption_ma_3`: 3-cycle rolling average of consumption
- `consumption_ma_5`: 5-cycle rolling average of consumption
- `consumption_trend`: Change in consumption from previous cycle
- `cumulative_consumption`: Total consumption up to current cycle
- `cycle_squared`: Quadratic cycle term for acceleration detection
- `cycle_log`: Logarithmic cycle term for plateau detection
- `weight_stress`: Interaction between weight and stress
- `age_cycle`: Interaction between age and cycle

### Output

- `consumption`: Predicted alcohol consumption in ml

## Model Performance

Typical performance metrics on test data:

- **R² Score**: ~0.96-0.97
- **Mean Absolute Error (MAE)**: ~0.15-0.25 ml
- **Root Mean Squared Error (RMSE)**: ~0.20-0.35 ml

Cross-validation R² scores typically range from 0.95 to 0.98.

## Visualizations

The model generates two main visualization files:

1. **mouse_alcohol_predictions.png**: Contains four subplots
   - Actual vs Predicted consumption scatter plot
   - Time series for a specific mouse with future predictions
   - Distribution of consumption across all mice
   - Average consumption by cycle across all mice

2. **feature_importance.png**: Bar chart showing the top 10 most important features

## Class Reference

### `MouseAlcoholPredictor`

Main class for alcohol consumption prediction.

#### Methods

- `generate_sample_data(n_mice=20, n_cycles=10)`: Generates synthetic mouse data
- `create_features(df)`: Engineers additional features from raw data
- `prepare_data(df, target_col='consumption')`: Prepares features and target for training
- `train(X_train, y_train, verbose=True)`: Trains the Random Forest model
- `predict(X)`: Makes predictions on new data
- `evaluate(X_test, y_test)`: Evaluates model performance
- `predict_future_cycles(mouse_data, n_future_cycles=5)`: Predicts future consumption
- `get_feature_importance()`: Returns feature importance scores

## Example Output

```
======================================================================
MOUSE ALCOHOL CONSUMPTION PREDICTION MODEL
======================================================================

Step 1: Generating sample data...
Generated data for 20 mice over 10 cycles
Total samples: 200

Step 2: Engineering features...
Created 14 features

Step 3: Preparing data for training...
Training samples: 160
Testing samples: 40

Step 4: Training model...
----------------------------------------------------------------------
Cross-validation R² scores: [0.977 0.965 0.976 0.967 0.952]
Mean CV R²: 0.9674 (+/- 0.0090)
----------------------------------------------------------------------

Step 5: Evaluating model on test data...
Mean Absolute Error: 0.1824 ml
Root Mean Squared Error: 0.2456 ml
R² Score: 0.9724

Step 6: Analyzing feature importance...

Top 10 Most Important Features:
              feature  importance
 previous_consumption    0.325412
   cumulative_consumption    0.186543
       consumption_ma_5    0.145621
       consumption_ma_3    0.112334
                 weight    0.087234
          cycle_squared    0.056712
                  cycle    0.034521
...

Step 7: Predicting future consumption...

Future predictions for Mouse 5:
   cycle  predicted_consumption
    11.0               1.527425
    12.0               1.548800
    13.0               1.526943
...
```

## Limitations

- Model is trained on synthetic data, so performance on real data may vary
- Future predictions assume stress levels remain within normal ranges
- Assumes consistent environmental conditions
