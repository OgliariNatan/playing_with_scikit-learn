"""
Exemplo de Previsão de Séries Temporais com ARIMA

Este script demonstra como usar ARIMA (AutoRegressive Integrated Moving Average)
para previsão de séries temporais. ARIMA é um método estatístico popular para análise
e previsão de dados de séries temporais.

Autor: Repositório Playing with scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def generate_sample_data(n_points=200):
    """
    Generate sample time series data with trend and seasonality
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    
    Returns:
    --------
    pd.Series : Time series data
    """
    np.random.seed(42)
    time = np.arange(n_points)
    
    # Create a time series with trend, seasonality, and noise
    trend = 0.5 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 5, n_points)
    
    data = trend + seasonality + noise + 50
    
    # Create a pandas Series with a date index
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    ts_data = pd.Series(data, index=dates)
    
    return ts_data


def split_time_series(data, test_size=0.2):
    """
    Split time series data into train and test sets
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    train, test : pd.Series
        Training and testing data
    """
    split_point = int(len(data) * (1 - test_size))
    train = data[:split_point]
    test = data[split_point:]
    return train, test


def fit_arima_model(train_data, order=(2, 1, 2)):
    """
    Fit an ARIMA model to the training data
    
    Parameters:
    -----------
    train_data : pd.Series
        Training time series data
    order : tuple
        ARIMA order (p, d, q)
        p: number of autoregressive terms
        d: number of differences
        q: number of moving average terms
    
    Returns:
    --------
    model_fit : ARIMAResults
        Fitted ARIMA model
    """
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit


def make_predictions(model_fit, steps):
    """
    Make predictions using the fitted ARIMA model
    
    Parameters:
    -----------
    model_fit : ARIMAResults
        Fitted ARIMA model
    steps : int
        Number of steps to forecast
    
    Returns:
    --------
    predictions : array
        Forecasted values
    """
    predictions = model_fit.forecast(steps=steps)
    return predictions


def evaluate_model(actual, predicted):
    """
    Evaluate model performance using various metrics
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    dict : Dictionary of evaluation metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }


def plot_results(train, test, predictions, title='ARIMA Forecasting Results'):
    """
    Plot the training data, test data, and predictions
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data
    predictions : array-like
        Predicted values
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, label='Training Data', color='blue')
    plt.plot(test.index, test.values, label='Test Data', color='green')
    plt.plot(test.index, predictions, label='ARIMA Predictions', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('arima_forecast.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'arima_forecast.png'")
    plt.show()


def main():
    """
    Main function to run the ARIMA forecasting example
    """
    print("=" * 60)
    print("ARIMA Time Series Forecasting Example")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample time series data...")
    data = generate_sample_data(n_points=200)
    print(f"   Generated {len(data)} data points")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # 2. Split data into train and test sets
    print("\n2. Splitting data into train and test sets...")
    train, test = split_time_series(data, test_size=0.2)
    print(f"   Training set size: {len(train)}")
    print(f"   Test set size: {len(test)}")
    
    # 3. Fit ARIMA model
    print("\n3. Fitting ARIMA model...")
    print("   Using ARIMA(2, 1, 2) - (p=2, d=1, q=2)")
    model_fit = fit_arima_model(train, order=(2, 1, 2))
    print("   Model fitted successfully!")
    
    # 4. Make predictions
    print("\n4. Making predictions on test set...")
    predictions = make_predictions(model_fit, steps=len(test))
    print(f"   Generated {len(predictions)} predictions")
    
    # 5. Evaluate model
    print("\n5. Evaluating model performance...")
    metrics = evaluate_model(test.values, predictions)
    print(f"   Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"   Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    print(f"   Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    
    # 6. Plot results
    print("\n6. Plotting results...")
    plot_results(train, test, predictions)
    
    print("\n" + "=" * 60)
    print("ARIMA forecasting example completed successfully!")
    print("=" * 60)


def exemplo_simples():
    """
    Exemplo simplificado de uso do ARIMA
    """
    print("\n" + "=" * 60)
    print("Exemplo Simples de ARIMA")
    print("=" * 60)
    
    # Criar dados de exemplo
    quantidade = pd.Series([10, 12, 15, 18, 20, 22, 25, 28, 30, 33])
    print(f"\nDados originais: {quantidade.values}")
    
    # Configurar e treinar o modelo ARIMA
    dados_serie_temporal = quantidade  # Exemplo de série
    modelo = ARIMA(dados_serie_temporal, order=(2, 1, 2))  # Parâmetros ajustáveis
    modelo_treinado = modelo.fit()
    
    # Fazer previsão para os próximos 3 pontos
    previsao = modelo_treinado.predict(start=len(dados_serie_temporal), end=len(dados_serie_temporal)+2)
    print(f"Previsão para os próximos 3 pontos: {previsao.values}")
    print("=" * 60)


def exemplo_mlp_regressor():
    """
    Exemplo de previsão usando Rede Neural MLP (Multi-Layer Perceptron)
    """
    from sklearn.neural_network import MLPRegressor
    
    print("\n" + "=" * 60)
    print("Exemplo de Previsão com MLP Regressor")
    print("=" * 60)
    
    # Dados de exemplo: anos e quantidade
    anos = np.array([[2015], [2016], [2017], [2018], [2019], [2020], [2021], [2022], [2023], [2024]])
    quantidade = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 33])
    
    print(f"\nDados de treinamento:")
    print(f"Anos: {anos.flatten()}")
    print(f"Quantidade: {quantidade}")
    
    # Configurar e treinar o modelo MLP
    modelo = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
    modelo.fit(anos, quantidade)
    
    # Fazer previsão para 2025
    previsao_2025 = modelo.predict([[2025]])
    print(f"\nPrevisão para 2025: {previsao_2025[0]:.2f}")
    
    # Fazer previsões para múltiplos anos
    anos_futuros = np.array([[2025], [2026], [2027]])
    previsoes = modelo.predict(anos_futuros)
    print(f"\nPrevisões para os próximos anos:")
    for ano, prev in zip(anos_futuros.flatten(), previsoes):
        print(f"  {ano}: {prev:.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Executar exemplo completo
    main()
    
    # Executar exemplo simples
    exemplo_simples()
    
    # Executar exemplo com MLP Regressor
    exemplo_mlp_regressor()
