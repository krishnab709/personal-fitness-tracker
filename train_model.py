import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Load data
calories = pd.read_csv("data/calories.csv")
exercise = pd.read_csv("data/exercise.csv")

# Merge and clean
df = exercise.merge(calories, on="User_ID")
df.drop(columns="User_ID", inplace=True)
df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
df["BMI"] = round(df["BMI"], 2)

# Select and preprocess features
df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Calories", axis=1)
y = df["Calories"]

# Train model
model = RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=3)
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/calorie_model.pkl")

print("✅ Model saved successfully as model/calorie_model.pkl")
