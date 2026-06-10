import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

st.title("📊 Student Performance Predictor")

studytime = st.number_input("Study Time", 1, 4)
failures = st.number_input("Failures", 0, 5)
absences = st.number_input("Absences", 0, 50)

input_df = pd.DataFrame([{
    "studytime": studytime,
    "failures": failures,
    "absences": absences
}])

if st.button("Predict"):
    result = model.predict(input_df)[0]
    st.success(f"Predicted Grade: {round(result,2)}")
