import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Personal Fitness Tracker", layout="centered")

# -------------------------------
# Load and Merge Datasets
# -------------------------------
@st.cache_data
def load_data():
    try:
        calories = pd.read_csv("data/calories.csv")
        exercise = pd.read_csv("data/exercise.csv")
        df = exercise.merge(calories, on="User_ID")
        df.drop(columns="User_ID", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# -------------------------------
# Add BMI and Preprocess
# -------------------------------
def preprocess(df):
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = round(df["BMI"], 2)
    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    df = pd.get_dummies(df, drop_first=True)
    return df

# -------------------------------
# Train Model
# -------------------------------
def train_model(df):
    X = df.drop("Calories", axis=1)
    y = df["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=3)
    model.fit(X_train, y_train)
    return model, X

# -------------------------------
# Sidebar User Input
# -------------------------------
def get_user_input(X_train_columns):
    st.sidebar.header("User Input Parameters:")

    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 22.0)
    st.sidebar.caption("💡 *BMI = Weight / (Height × Height)*")
    duration = st.sidebar.slider("Workout Duration (min)", 0, 60, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 150, 85)
    body_temp = st.sidebar.slider("Body Temp (°C)", 36.0, 42.0, 38.0)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])

    gender_male = 1 if gender == "Male" else 0

    user_data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_male
    }

    df = pd.DataFrame(user_data, index=[0])
    df = df.reindex(columns=X_train_columns, fill_value=0)  # match training data columns
    return df

# -------------------------------
# App Start
# -------------------------------
st.title("🏋️ Personal Fitness Tracker")
st.write("Enter your fitness metrics to predict estimated calories burned using a trained ML model.")

# Load and process data
raw_data = load_data()
if raw_data is not None:
    data = preprocess(raw_data)
    model, X_template = train_model(data)
    user_df = get_user_input(X_template.columns)

    st.write("### Your Input Summary")
    st.dataframe(user_df)

    with st.spinner("Running prediction..."):
        time.sleep(1.5)
        prediction = model.predict(user_df)[0]
        st.success(f"🔥 Estimated Calories Burned: **{round(prediction, 2)} kcal**")

    # Show similar data
    st.write("---")
    st.subheader("📊 Similar Past Records")
    cal_range = [prediction - 10, prediction + 10]
    similar = raw_data[(raw_data["Calories"] >= cal_range[0]) & (raw_data["Calories"] <= cal_range[1])]
    st.dataframe(similar.sample(min(5, len(similar))))

    # General Comparison
    st.write("---")
    st.subheader("📈 General Info Compared to Others")

    st.write(f"You are older than **{round((raw_data['Age'] < user_df['Age'][0]).mean() * 100, 2)}%** of users.")
    st.write(f"Your workout duration is longer than **{round((raw_data['Duration'] < user_df['Duration'][0]).mean() * 100, 2)}%** of users.")
    st.write(f"Your heart rate is higher than **{round((raw_data['Heart_Rate'] < user_df['Heart_Rate'][0]).mean() * 100, 2)}%** of users.")
    st.write(f"Your body temperature is higher than **{round((raw_data['Body_Temp'] < user_df['Body_Temp'][0]).mean() * 100, 2)}%** of users.")

else:
    st.stop()
