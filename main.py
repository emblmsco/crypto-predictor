import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import time
import os
from scipy.stats import norm
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines (à ajuster en production)
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

# 1. Récupération des données en temps réel
def fetch_real_time_data(symbol, interval='15m', limit=2000):
    attempt = 0
    while attempt < 5:
        try:
            print(f"📡 Récupération des données pour {symbol} (tentative {attempt + 1})...")
            url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                print(f"✅ Données récupérées avec succès ({len(df)} lignes).")
                return df
            else:
                print(f"❌ Erreur lors de la récupération des données : {response.status_code}")
                attempt += 1
                time.sleep(5)
        except Exception as e:
            print(f"⚠️ Erreur inattendue lors de la récupération des données : {str(e)}")
            time.sleep(5)
            attempt += 1
    raise Exception("❌ Impossible de récupérer les données après 5 tentatives.")

# 2. Ajout des indicateurs techniques
def add_technical_indicators(df):
    print("📊 Ajout des indicateurs techniques...")
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['EMA_20'] = ta.ema(df['close'], length=20)
    bbands = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if 'STOCHk_14_3_3' in stoch.columns and 'STOCHd_14_3_3' in stoch.columns:
        df['STOCH_k'] = stoch['STOCHk_14_3_3']
        df['STOCH_d'] = stoch['STOCHd_14_3_3']
    else:
        print("⚠️ Les colonnes STOCHk et STOCHd sont manquantes dans le DataFrame stoch.")
    df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    df.dropna(inplace=True)
    print(f"✅ Indicateurs techniques ajoutés avec succès. Nombre de lignes après nettoyage: {len(df)}")
    return df

# 3. Préparation des données pour le modèle
def prepare_data(df, time_step=60):
    print("📦 Préparation des données pour le modèle...")
    expected_columns = ['close', 'RSI', 'MACD_12_26_9', 'SMA_20', 'EMA_20', 'ATR', 'STOCH_k', 'STOCH_d', 'CCI']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler_X.fit_transform(df[expected_columns])
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_y = scaler_y.fit_transform(df[['close']])
    print(f"Nombre d'échantillons après normalisation: {len(scaled_X)}")
    X, y = [], []
    for i in range(time_step, len(scaled_X)):
        X.append(scaled_X[i-time_step:i])
        y.append(scaled_y[i, 0])
    print(f"Nombre d'échantillons après préparation: {len(X)}")
    return np.array(X), np.array(y), scaler_X, scaler_y

# 4. Création du modèle LSTM
def create_model(input_shape):
    print("🛠️ Création du modèle LSTM...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    print("✅ Modèle LSTM créé avec succès.")
    return model

# 5. Sauvegarde et chargement du modèle
def load_or_train_model(X_train, y_train):
    if os.path.exists("model_lstm.h5"):
        print("⚡ Chargement du modèle préexistant...")
        model = load_model("model_lstm.h5")
    else:
        model = create_model((X_train.shape[1], X_train.shape[2]))
        print("🛠️ Entraînement du modèle en cours...")
        model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=2)
        model.save("model_lstm.h5")
    return model

# 6. Calcul de la probabilité
def calculate_probability(predicted_price, actual_price, volatility=0.05):  # Volatilité augmentée
    std_dev = actual_price * volatility
    probability = norm.cdf(predicted_price, loc=actual_price, scale=std_dev)
    probability = max(probability, 0.01)  # Probabilité minimale de 1%
    return float(probability)

# 7. Stratégie gagnante
def trading_strategy(predicted_price, actual_price, probability, threshold=0.7):  # Seuil réduit
    if probability > threshold:
        if predicted_price > actual_price:
            return "Achat"
        else:
            return "Vente"
    else:
        return "Attendre"

# 8. Endpoint pour la prédiction
@app.get("/predict/{symbol}", summary="Prédire le prix d'une cryptomonnaie", description="Cette API prédit le prix d'une cryptomonnaie en temps réel.")
def predict(symbol: str, interval: str = '15m', limit: int = 2000):
    try:
        df = fetch_real_time_data(symbol, interval, limit)
        df = add_technical_indicators(df)
        if len(df) < 60:
            raise HTTPException(status_code=400, detail="Pas assez de données après traitement.")
        X_train, y_train, scaler_X, scaler_y = prepare_data(df)
        if len(X_train) < 60:
            raise HTTPException(status_code=400, detail="Pas assez de données pour entraîner le modèle.")
        model = load_or_train_model(X_train, y_train)
        scaled_X_new = scaler_X.transform(df[['close', 'RSI', 'MACD_12_26_9', 'SMA_20', 'EMA_20', 'ATR', 'STOCH_k', 'STOCH_d', 'CCI']])
        if len(scaled_X_new) < 60:
            raise HTTPException(status_code=400, detail="Pas assez de données pour la prédiction.")
        X_new = np.array([scaled_X_new[-60:, :]])
        X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 9))
        prediction = model.predict(X_new)
        predicted_price = scaler_y.inverse_transform(prediction)
        actual_price = df['close'].iloc[-1]
        change = (predicted_price[0][0] - actual_price) / actual_price
        probability = calculate_probability(predicted_price[0][0], actual_price)
        action = trading_strategy(predicted_price[0][0], actual_price, probability)

        # Convertir toutes les valeurs numpy en types natifs Python
        predicted_price = round(float(predicted_price[0][0]), 2)  # 2 décimales
        actual_price = round(float(actual_price), 2)  # 2 décimales
        change = round(float(change), 2)  # 4 décimales
        probability = round(float(probability), 2)  # 4 décimales

        # Ajouter des logs pour vérifier les types de données
        print(f"Type de predicted_price : {type(predicted_price)}")  # Doit être <class 'float'>
        print(f"Type de actual_price : {type(actual_price)}")  # Doit être <class 'float'>
        print(f"Type de change : {type(change)}")  # Doit être <class 'float'>
        print(f"Type de probability : {type(probability)}")  # Doit être <class 'float'>

        return {
            "symbol": symbol,
            "predicted_price": predicted_price,
            "actual_price": actual_price,
            "change": change,
            "probability": probability,
            "action": action
        }
    except Exception as e:
        print(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/updates/{symbol}")
async def updates(symbol: str):
    async def event_stream():
        while True:
            try:
                # Récupérer les données en temps réel
                df = fetch_real_time_data(symbol, interval='15m', limit=2000)
                df = add_technical_indicators(df)
                if len(df) < 60:
                    continue  # Pas assez de données, passer à l'itération suivante

                # Préparer les données pour la prédiction
                X_train, y_train, scaler_X, scaler_y = prepare_data(df)
                if len(X_train) < 60:
                    continue  # Pas assez de données, passer à l'itération suivante

                # Charger ou entraîner le modèle
                model = load_or_train_model(X_train, y_train)

                # Faire une prédiction
                scaled_X_new = scaler_X.transform(df[['close', 'RSI', 'MACD_12_26_9', 'SMA_20', 'EMA_20', 'ATR', 'STOCH_k', 'STOCH_d', 'CCI']])
                if len(scaled_X_new) < 60:
                    continue  # Pas assez de données, passer à l'itération suivante

                X_new = np.array([scaled_X_new[-60:, :]])
                X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 9))
                prediction = model.predict(X_new)
                predicted_price = scaler_y.inverse_transform(prediction)
                actual_price = df['close'].iloc[-1]
                change = (predicted_price[0][0] - actual_price) / actual_price
                probability = calculate_probability(predicted_price[0][0], actual_price)
                action = trading_strategy(predicted_price[0][0], actual_price, probability)

                # Convertir les résultats en JSON
                result = {
                    "symbol": symbol,
                    "predicted_price": float(predicted_price[0][0]),
                    "actual_price": float(actual_price),
                    "change": float(change),
                    "probability": float(probability),
                    "action": action,
                    "historical_data": {
                        "labels": df.index[-60:].astype(str).tolist(),  # Timestamps
                        "prices": df['close'][-60:].tolist()  # Prix de clôture
                    }
                }

                # Envoyer les données au client
                yield f"data: {json.dumps(result)}\n\n"
            except Exception as e:
                print(f"Erreur lors de la prédiction : {str(e)}")
                break

            # Attendre avant la prochaine mise à jour
            await asyncio.sleep(60)  # Mettre à jour toutes les 60 secondes

    return StreamingResponse(event_stream(), media_type="text/event-stream")
            
# Pour exécuter l'API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)