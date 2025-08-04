# Stock-Price-Prediction

This project presents an interactive web application that predicts future stock prices of publicly traded companies using historical stock market data. The application is powered by a deep learning model trained on time-series stock data and is deployed using Streamlit.

---

## Project Overview

The application uses the **LSTM (Long Short-Term Memory)** neural network to predict stock prices based on historical data fetched from Yahoo Finance. It provides visualizations of actual vs predicted prices and various moving averages like 50, 100, and 200 days.

---

## Technologies Used

| Technology        | Description                                  |
|-------------------|----------------------------------------------|
| Python            | Core programming language                    |
| Keras             | For building and loading the LSTM model      |
| Streamlit         | For building the web interface               |
| yfinance          | To download stock market data                |
| scikit-learn      | For scaling and preprocessing data           |
| matplotlib        | For data visualization                       |
| NumPy / Pandas    | For numerical and data manipulation          |

---

## Project Structure
stock-price-prediction/

├── app.py # Streamlit web app

├── Stock Price Prediction.ipynb # Notebook to train/test model

├── stock prediction-checkpoint.ipynb # Auto-saved Jupyter checkpoint

├── Stock Predictions Model.keras # Pre-trained LSTM model

└── README.md # Project documentation


---

## Features

- **Real-time stock data**: Users can input any valid stock symbol (e.g., AAPL, GOOG) and get historical stock data from 2014 to 2025.
- **LSTM-based prediction**: A deep learning model trained on historical price data predicts the future stock price.
- **Visual Analysis**:
  - Price vs Moving Average (50 days)
  - Price vs MA 50 & MA 100
  - Price vs MA 100 & MA 200
- **Comparison**: Actual vs Predicted stock prices plotted side-by-side.

---

## How it Works

1. **User Input**: Enter a stock ticker symbol (e.g., GOOG).
2. **Data Fetching**: Stock data is fetched from Yahoo Finance from `2014-01-01` to `2025-03-31`.
3. **Data Preprocessing**:
   - Split into train and test (80%-20%)
   - Apply MinMax scaling
   - Extract rolling data of 100 previous days
4. **Prediction**: The trained LSTM model predicts the stock price on test data.
5. **Output**: The app shows a visual comparison of actual vs predicted prices.

---

## Screenshots
### Moving average  
![Moving Average Visualization](https://github.com/Ravikumar0311/Stock-Price-Prediction/blob/main/visuals/moving%20average.png)
### Predicted VS actual price
![Prediction](https://github.com/Ravikumar0311/Stock-Price-Prediction/blob/main/visuals/actual%20vs%20predicted%20price.png)

---

## Limitations
- Predictive accuracy depends on training data and market volatility.
- Only works with stocks supported by Yahoo Finance.
- No advanced technical indicators (e.g., RSI, MACD) used yet.

## License
This project is licensed under the MIT License.
