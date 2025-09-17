# Databricks notebook source
# MAGIC %restart_python

# COMMAND ----------

from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .remote("")   # remove Spark Connect remote
         .getOrCreate())

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df_features = spark.read.table("workspace.default.features")
df_train = spark.read.table("workspace.default.train")
df_stores = spark.read.table("workspace.default.stores")
df_partial = df_train.join(df_features, on=["Store", "Date", "IsHoliday"], how="left")
df = df_partial.join(df_stores, on="Store", how="left") 
df.show(5)

# COMMAND ----------

df.printSchema()
df.count()

# COMMAND ----------

bronze_df_new_full = spark.read.table("workspace.default.Full_df")
bronze_df_new_full.show(5)

# COMMAND ----------

from pyspark.sql.functions import col,sum,when

# COMMAND ----------

null_counts = bronze_df_new_full.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in bronze_df_new_full.columns
])
null_counts.show()

# COMMAND ----------

from pyspark.sql.types import StringType

null_counts = bronze_df_new_full.select([
    sum(
        (
            col(c).isNull() if not isinstance(bronze_df_new_full.schema[c].dataType, StringType)
            else (col(c).isNull() | col(c).isin(['NA', '']))
        ).cast("int")
    ).alias(c)
    for c in bronze_df_new_full.columns])
display(null_counts)

# COMMAND ----------

problem_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
                "CPI","Unemployment"]

# COMMAND ----------

for c in problem_cols:
    bronze_df = bronze_df_new_full.withColumn(c, when(col(c) == "NA", None).otherwise(col(c)))

# COMMAND ----------

from pyspark.sql.functions import lower
silver_df_new_full = (bronze_df_new_full
             .withColumn("Store", col("Store").cast("int"))
             .withColumn("Dept", col("Dept").cast("int"))
             .withColumn("Date", col("Date").cast("date"))
             .withColumn("Weekly_Sales", col("Weekly_Sales").cast("double"))
             .withColumn("Temperature", col("Temperature").cast("double"))
             .withColumn("Fuel_Price", col("Fuel_Price").cast("double"))
             .withColumn("CPI", col("CPI").cast("double"))
             .withColumn("Unemployment", col("Unemployment").cast("double"))
             .withColumn("IsHoliday", lower(col("IsHoliday")).cast("boolean"))
            )

# COMMAND ----------

silver_df_new_full = silver_df_new_full.fillna({c: 0 for c in ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]})

# COMMAND ----------

silver_df_new_full.show(20)

# COMMAND ----------

silver_df_new_full.columns

# COMMAND ----------

spark


# COMMAND ----------

pandas_df = df.sample(fraction=0.05, seed=42).toPandas()
print(pandas_df.shape)
pandas_df.head()

# COMMAND ----------

import numpy as np

# Replace "NA" with np.nan
pandas_df.replace("NA", np.nan, inplace=True)

# Fill missing numeric values with 0 (or median if you prefer)
for col in ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment"]:
    pandas_df[col] = pd.to_numeric(pandas_df[col], errors="coerce").fillna(0)

# Ensure correct dtypes
pandas_df["IsHoliday"] = pandas_df["IsHoliday"].astype(int)  # 0/1
pandas_df["Weekly_Sales"] = pd.to_numeric(pandas_df["Weekly_Sales"], errors="coerce")


# COMMAND ----------

# Extract date features
pandas_df["Date"] = pd.to_datetime(pandas_df["Date"])
pandas_df["Year"] = pandas_df["Date"].dt.year
pandas_df["Month"] = pandas_df["Date"].dt.month
pandas_df["WeekOfYear"] = pandas_df["Date"].dt.isocalendar().week
pandas_df["DayOfWeek"] = pandas_df["Date"].dt.dayofweek


# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Features & target
target = "Weekly_Sales"
features = ["Temperature","Fuel_Price","CPI","Unemployment",
            "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
            "Size","IsHoliday","Type","Year","Month","WeekOfYear","DayOfWeek"]

X = pandas_df[features]
y = pandas_df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = ["Temperature","Fuel_Price","CPI","Unemployment",
                    "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
                    "Size","Year","Month","WeekOfYear","DayOfWeek"]

categorical_features = ["Type","IsHoliday"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ML pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                       ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))


# COMMAND ----------

# MAGIC %pip install xgboost
# MAGIC %pip install prophet
# MAGIC %pip install tensorflow
# MAGIC

# COMMAND ----------

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Use the same preprocessor from before
xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("XGBoost R²:", r2_score(y_test, y_pred_xgb))


# COMMAND ----------

from prophet import Prophet

# Filter one store
store_df = pandas_df[pandas_df["Store"] == 1][["Date", "Weekly_Sales"]].copy()
store_df = store_df.groupby("Date", as_index=False).sum()  # aggregate by date
store_df.columns = ["ds", "y"]  # Prophet requires these names

# Train Prophet
prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet.fit(store_df)

# Forecast next 90 days
future = prophet.make_future_dataframe(periods=90, freq="D")
forecast = prophet.predict(future)

# Plot
fig = prophet.plot(forecast)


# COMMAND ----------

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Example: one store’s weekly sales
store_df = pandas_df[pandas_df["Store"] == 1][["Date","Weekly_Sales"]].copy()
store_df = store_df.groupby("Date", as_index=False).sum()

# Scale sales
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(store_df[["Weekly_Sales"]])

# Create sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 10
X, y = create_sequences(sales_scaled, SEQ_LEN)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM
model = Sequential([
    LSTM(64, activation="relu", input_shape=(SEQ_LEN, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)

print("Sample predictions:", y_pred_rescaled[:5].flatten())


# COMMAND ----------

import matplotlib.pyplot as plt

# Rescale actual values
y_test_rescaled = scaler.inverse_transform(y_test)

plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label="Actual")
plt.plot(y_pred_rescaled, label="Predicted")
plt.legend()
plt.title("LSTM Forecast - Store 1")
plt.show()


# COMMAND ----------

