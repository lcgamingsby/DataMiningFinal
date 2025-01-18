import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import streamlit as st

# Streamlit App Title
st.title("Exchange Rate Prediction")

# Step 1: Input for Base and Target Currencies
st.sidebar.header("Currency Selection")
base_currency = st.sidebar.text_input("Base Currency (e.g., USD):", "USD")
target_currency = st.sidebar.text_input("Target Currency (e.g., EUR):", "EUR")

# Step 2: Fetch Exchange Rate Data
@st.cache_data
def fetch_exchange_rate(api_url, api_key, base_currency, target_currency, start_date, end_date):
    response = requests.get(
        f"{api_url}/timeseries",
        params={
            "start_date": start_date,
            "end_date": end_date,
            "base": base_currency.upper(),
            "symbols": target_currency.upper(),
            "apikey": api_key
        }
    )
    if response.status_code == 200:
        data = response.json()
        rates = data.get("rates", {})
        return pd.DataFrame(
            [{"date": date, "rate": rates[date][target_currency.upper()]} for date in rates]
        )
    else:
        st.error("Failed to fetch data. Please check your inputs and API key.")
        return None

# API Setup
api_url = "https://api.apilayer.com/exchangerates_data"
api_key = st.secrets["API_KEY"]  # Use Streamlit Secrets for security
end_date = datetime.now().date()
start_date = (end_date - timedelta(days=365)).strftime("%Y-%m-%d")

if base_currency and target_currency:
    df = fetch_exchange_rate(api_url, api_key, base_currency, target_currency, start_date, end_date)

    if df is not None:
        # Data Cleaning
        def clean_data(data):
            data["date"] = pd.to_datetime(data["date"])
            data.sort_values(by="date", inplace=True)
            return data

        df = clean_data(df)

        # Exploratory Data Analysis (EDA)
        def exploratory_data_analysis(data):
            st.subheader("Data Summary")
            st.write(data.describe())

            st.subheader("Exchange Rate Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data["rate"], kde=True, bins=30, ax=ax, color="green")
            st.pyplot(fig)

        exploratory_data_analysis(df)

        # Feature Engineering
        def create_features(data):
            data["year"] = data["date"].dt.year
            data["month"] = data["date"].dt.month
            data["day"] = data["date"].dt.day
            return data

        df = create_features(df)

        # Define features and target
        X = df[["year", "month", "day"]]
        y = df["rate"]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Regression Model
        rf = RandomForestRegressor(random_state=42)
        param_grid = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluate Model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R^2 Score: {r2}")

        # Prediction for the Next Month
        def predict_next_month(model, last_date, months=1):
            future_dates = []
            for month in range(1, months + 1):
                future_date = last_date + timedelta(days=30 * month)
                future_dates.append(future_date)

            future_features = pd.DataFrame({
                "year": [date.year for date in future_dates],
                "month": [date.month for date in future_dates],
                "day": [date.day for date in future_dates]
            })

            future_predictions = model.predict(future_features)
            return future_dates, future_predictions

        last_date = df["date"].max()
        future_dates, future_predictions = predict_next_month(best_model, last_date)

        st.subheader(f"Predicted Exchange Rates for {base_currency} to {target_currency}")
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Rate": future_predictions})
        st.write(future_df)

        # Visualization
        st.subheader("Exchange Rate Prediction Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["date"], best_model.predict(X), label="Predicted Rates", color="red", linewidth=2)
        ax.scatter(future_dates, future_predictions, color="green", label="Predicted Next Month", marker="x")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Exchange Rate ({base_currency} to {target_currency})")
        ax.set_title(f"Exchange Rate Prediction: {base_currency} to {target_currency}")
        ax.legend()
        st.pyplot(fig)
