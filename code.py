import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (assuming a CSV file)
data = pd.read_csv("credit_card_transactions.csv")

# Display basic info
print(data.head())
print(data.info())

# Convert timestamp column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Visualize transaction trends
data['amount'].plot(figsize=(12, 6))
plt.title("Transaction Amount Over Time")
plt.show()

# Perform Seasonal Decomposition
decomposition = seasonal_decompose(data['amount'], model='additive', period=30)
decomposition.plot()
plt.show()

# Compute ACF and PACF
lag_acf = acf(data['amount'].dropna(), nlags=20)
lag_pacf = pacf(data['amount'].dropna(), nlags=20)

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.stem(lag_acf)
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.stem(lag_pacf)
plt.title('Partial Autocorrelation Function')
plt.show()

# Spectral Analysis
frequencies, spectrum = periodogram(data['amount'].dropna())
plt.figure(figsize=(10, 5))
plt.plot(frequencies, spectrum)
plt.title('Spectral Analysis')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()

# Feature Engineering
data['hour'] = data.index.hour
data['day'] = data.index.day
data['month'] = data.index.month
data['day_of_week'] = data.index.dayofweek

# Define features and target
X = data[['amount', 'hour', 'day', 'month', 'day_of_week']]
y = data['is_fraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
