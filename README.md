# playing_with_scikit-learn
Para nocões do scikit-learn

## Previsão de Séries Temporais com ARIMA

Este repositório inclui um exemplo de previsão de séries temporais com ARIMA (AutoRegressive Integrated Moving Average) usando Python.

### O que é ARIMA?

ARIMA é um método estatístico popular para análise e previsão de dados de séries temporais. Ele combina:
- **AR (AutoRegressivo)**: Usa valores passados para prever valores futuros
- **I (Integrado)**: Diferencia os dados para torná-los estacionários
- **MA (Média Móvel)**: Usa erros de previsão passados para prever valores futuros

### Instalação

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

### Executando o Exemplo

Para executar o exemplo de previsão ARIMA:

```bash
python arima_example.py
```

Isso irá:
1. Gerar dados de série temporal de amostra com tendência e sazonalidade
2. Dividir os dados em conjuntos de treinamento e teste
3. Ajustar um modelo ARIMA(2,1,2) aos dados de treinamento
4. Fazer previsões no conjunto de teste
5. Avaliar o desempenho do modelo usando MSE, RMSE e MAE
6. Gerar uma visualização salva como `arima_forecast.png`

### Arquivos

- `arima_example.py`: Script principal do exemplo de previsão ARIMA
- `requirements.txt`: Dependências de pacotes Python
- `arima_forecast.png`: Visualização gerada (após executar o exemplo)
