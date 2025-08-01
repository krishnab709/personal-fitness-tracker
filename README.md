# 🏋️ Personal Fitness Tracker – Streamlit Web App

An interactive machine learning-based fitness tracker that helps users estimate calories burned during workouts based on personal health metrics. Built with **Streamlit**, powered by a **Random Forest model**, and visualized with real-time feedback and comparisons.

---

## 🚀 Features

-  Predicts **calories burned** using:
  - Age
  - BMI
  - Gender
  - Workout Duration
  - Heart Rate
  - Body Temperature

- 🧠 **Machine Learning** with Random Forest Regressor
- 📊 **Real-time visualization** of similar past workout records
- 📈 Comparison of user input with historical data (age, duration, etc.)
- 🧮 Built-in **BMI calculator** with explanation
- 🌐 **Streamlit UI** with progress bars and intuitive inputs

---

##  Technologies Used

- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Joblib (for loading the ML model)

---

##  Project Structure


personal-fitness-tracker/
├── app.py # Streamlit application
├── requirements.txt # List of dependencies
├── model/
│ └── calorie_model.pkl # Pre-trained Random Forest model
├── data/
│ ├── calories.csv # Calories burned dataset
│ └── exercise.csv # Exercise details dataset

👤 Author
Krishna Bachhao
Final Year B.E. IT Student - SPPU
