# playing_with_scikit-learn
Para nocões do scikit-learn

## ARIMA Time Series Forecasting

This repository includes an example of ARIMA (AutoRegressive Integrated Moving Average) time series forecasting using Python.

### What is ARIMA?

ARIMA is a popular statistical method for analyzing and forecasting time series data. It combines:
- **AR (AutoRegressive)**: Uses past values to predict future values
- **I (Integrated)**: Differences the data to make it stationary
- **MA (Moving Average)**: Uses past forecast errors to predict future values

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Example

To run the ARIMA forecasting example:

```bash
python arima_example.py
```

This will:
1. Generate sample time series data with trend and seasonality
2. Split the data into training and test sets
3. Fit an ARIMA(2,1,2) model to the training data
4. Make predictions on the test set
5. Evaluate the model performance using MSE, RMSE, and MAE
6. Generate a visualization saved as `arima_forecast.png`

### Files

- `arima_example.py`: Main ARIMA forecasting example script
- `requirements.txt`: Python package dependencies
- `arima_forecast.png`: Generated visualization (after running the example)
