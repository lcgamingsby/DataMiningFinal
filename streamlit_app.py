import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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

df = fetch_exchange_rate(api_url, api_key, base_currency, target_currency, start_date, end_date)

if base_currency and target_currency:
    df = fetch_exchange_rate(api_url, api_key, base_currency, target_currency, start_date, end_date)
    
    if df is not None:
        # Data Cleaning
        def clean_data(data):
            data["date"] = pd.to_datetime(data["date"])
            data.sort_values(by="date", inplace=True)
            return data

        df = clean_data(df)

# EDA
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

# Clustering - Elbow Method
st.subheader("Clustering Analysis")
X_cluster = df[['rate']]
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
st.pyplot(plt)

# Apply optimal clustering
df['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_cluster)
st.write(df.head())

# Define features and target for regression
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

# Classification Model
df['label'] = (df['rate'].diff() > 0).astype(int)
X_class = df[['year', 'month', 'day']]
y_class = df['label']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)
st.subheader("Classification Report")
st.text(classification_report(y_test_cls, y_pred_cls))

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Visualization
st.subheader("Exchange Rate Prediction Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["date"], best_model.predict(X), label="Predicted Rates", color="red", linewidth=2)

# Step 1: Generate Future Dates (Next 30 Days)
future_dates = pd.date_range(df["date"].max() + timedelta(days=1), periods=30).strftime('%Y-%m-%d')

# Step 2: Feature Engineering for Future Dates
future_df = pd.DataFrame(future_dates, columns=["date"])
future_df["date"] = pd.to_datetime(future_df["date"])
future_df["year"] = future_df["date"].dt.year
future_df["month"] = future_df["date"].dt.month
future_df["day"] = future_df["date"].dt.day

# Step 3: Make Predictions for Future Dates
future_X = future_df[["year", "month", "day"]]
future_predictions = best_model.predict(future_X)

# Plot the historical data and predictions
ax.plot(future_df["date"], future_predictions, label="Predicted Rates (Next Month)", marker="o", color="blue", markersize=5, linestyle="-")

# Labels and title
ax.set_xlabel("Date")
ax.set_ylabel(f"Exchange Rate ({base_currency} to {target_currency})")
ax.set_title(f"Exchange Rate Prediction: {base_currency} to {target_currency}")

# Legend and grid
ax.legend()
ax.grid(True)

# Show the plot in Streamlit
st.pyplot(fig)
