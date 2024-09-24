import streamlit as st
import joblib 
import numpy as np

st.title('Employee attrition prediction result')

try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'random_forest_model.pkl' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

input_data = st.session_state.get('input_data',None)

if input_data:
    data_to_predict = np.array([
        input_data['Age'],input_data['DailyRate'],input_data['Department'],input_data['DistanceFromHome'],input_data['EnvironmentSatisfaction'],input_data['Gender'],input_data['JobLevel'],input_data['JobRole'],input_data['JobSatisfaction'],
        input_data['JobSatisfaction'],input_data['MonthlyIncome'],input_data['MonthlyRate'],input_data['NumCompaniesWorked'],input_data['OverTime'],input_data['TotalWorkingYears'],input_data['WorkLifeBalance'],input_data['YearsAtCompany'],input_data['YearsInCurrentRole'],input_data['YearsSinceLastPromotion']
    ]).reshape(1,-1)

    prediction = model.predict(data_to_predict)[0]

    if prediction == 1:
        st.success('This emplyee is likely to leave')
    else:
        st.success('This employee is likely to stay')
else:
    st.error('No input data found! Please refill correctly')