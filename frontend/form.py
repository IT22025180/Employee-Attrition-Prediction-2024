import streamlit as st
import numpy as np
from urllib.parse import urlparse,parse_qs
import joblib
import pandas as pd

st.set_page_config(page_title="Employee Attrition Predictor 2024", layout="wide")

def load_model():
    try:
        return joblib.load('classification_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'random_forest_model.pkl' is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# Load the model
model = load_model()

def get_query_params():
    
    query_params = st.query_params
    return query_params.get("page", ["input"])[0]

def display_page():
    current_page = get_query_params()

    if current_page == 'input':
        st.title('Employee Attrition Predictor')
        input_data = user_input_form()
        if st.button('Predict'):
            st.session_state['input_data'] = input_data
            # st.experimental_set_query_params(page='result')
   
    elif current_page == 'result':
        import result

def user_input_form():
    with st.form("input_form"):
        st.title("Employee Attrition Predictor")
        Age = st.selectbox(
            'Choose your age:',
            ['18', '19', '20', '21', '22','23','24','25','26','27','28',
            '29','30','31','32','33','34','35','36','37','38',
            '39','40','41','42','43','44','45','46','47','48',
            '49','50','51','52','53','54','55','56','57','58','59','60']
        )
        

        DailyRate = st.number_input(
            'Enter your daily rate:',
            min_value=0,      # Minimum value allowed
            key='dailyRate'
        )
        

        Department = st.selectbox(
            'Choose your department:',
            ['Sales', 'Research & Development', 'Human Resources']
        )
        if Department == 'Sales':
            Department = 2
        elif Department == 'Research & Development':
            Department = 1
        elif Department == 'Human Resources':
            Department = 0
        


        DistanceFromHome = st.number_input(
            'Enter distance in KM:',
            min_value=0,      # Minimum value allowed
            max_value=100,    # Maximum value allowed
            step=1,            # Step size
            key='distfromhome'
        )
        

        EnvironmentSatisfaction = st.selectbox(
            'Choose your environment satisfaction:',
            ['4', '3', '2', '1']
        )
        

        Gender = st.selectbox(
            'Gender:',
            ['Male', 'Female']
        )
        if Gender == 'Male':
            Gender = 1
        elif Gender == 'Female':
            Gender = 0
        

        JobLevel = st.selectbox(
            'Choose your job level:',
            ['5', '4', '3', '2', '1']
        )
        

        JobRole = st.selectbox(
            'Choose your job role:',
            ['Human Resources', 'Research Director', 'Sales Representative', 'Manager', 
             'Healthcare Representative', 'Manufacturing Director', 'Laboratory Technician', 
             'Research Scientist', 'Sales Executive']
        )
        if JobRole == 'Human Resources':
            JobRole = 8
        elif JobRole == 'Research Director':
            JobRole = 7
        elif JobRole == 'Sales Representative':
            JobRole = 6
        elif JobRole == 'Manager':
            JobRole = 5
        elif JobRole == 'Healthcare Representative':
            JobRole = 4
        elif JobRole == 'Manufacturing Director':
            JobRole = 3
        elif JobRole == 'Laboratory Technician':
            JobRole = 2
        elif JobRole == 'Research Scientist':
            JobRole = 1
        elif JobRole == 'Sales Executive':
            JobRole = 0
        

        JobSatisfaction = st.selectbox(
            'Choose your job satisfaction:',
            ['4', '3', '2', '1']
        )
        

        MaritalStatus = st.selectbox(
            'Choose your marital status:',
            ['Married', 'Divorced', 'Single']
        )
        if MaritalStatus == 'Married':
            MaritalStatus = 2
        elif MaritalStatus == 'Divorced':
            MaritalStatus = 1
        elif MaritalStatus == 'Single':
            MaritalStatus = 0
        

        MonthlyIncome = st.number_input(
            'Enter your monthly income:',
            min_value=0,      # Minimum value allowed
            key = 'monthlyincome'
        )
        

        MonthlyRate = st.number_input(
            'Enter your monthly rate:',
            min_value=0,      # Minimum value allowed
            key = 'monthlyrate'
        )
        

        NumCompaniesWorked = st.selectbox(
            'Number of companies work with:',
            ['1', '2', '3', '4', '5','6','7','8','9','10']
        )
        

        OverTime = st.selectbox(
            'Do you work overtime?',
            ['Yes', 'No']
        )
        if OverTime == 'Yes':
            OverTime = 1
        elif OverTime == 'No':
            OverTime = 0
        

        TotalWorkingYears = st.number_input(
            'Total working years:',
            min_value=0,      # Minimum value allowed
            max_value=20,    # Maximum value allowed
            step=1,            # Step size
            key='totalworkyrs'
        )
        

        WorkLifeBalance = st.selectbox(
            'Choose your work life balance:',
            ['4', '3', '2', '1']
        )
        

        YearsAtCompany = st.number_input(
            'Years at company:',
            min_value=0,      # Minimum value allowed
            max_value=20,    # Maximum value allowed
            step=1,         # Step size
            key='yrsAtcompany'
        )
        

        YearsInCurrentRole = st.selectbox(
            'Choose your years in current role:',
            ['0', '1', '2', '3', '4', '5','6','7','8','9','10',
            '11','12','13','14','15','16','17','18','19','20']
        )
        

        YearsSinceLastPromotion = st.selectbox(
            'Choose your year since last promotion:',
            ['0', '1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15']
        )
        

        submitted = st.form_submit_button("Predict")

    if submitted:

        input_data = {
            'Age': Age, 'DailyRate': DailyRate, 'Department': Department, 'DistanceFromHome': DistanceFromHome,
            'EnvironmentSatisfaction': EnvironmentSatisfaction, 'Gender': Gender, 'JobLevel': JobLevel, 'JobRole': JobRole,
            'JobSatisfaction': JobSatisfaction, 'MaritalStatus': MaritalStatus,'MonthlyIncome': MonthlyIncome, 'MonthlyRate': MonthlyRate,
            'NumCompaniesWorked': NumCompaniesWorked, 'OverTime': OverTime, 'TotalWorkingYears': TotalWorkingYears, 'WorkLifeBalance': WorkLifeBalance,
            'YearsAtCompany': YearsAtCompany, 'YearsInCurrentRole': YearsInCurrentRole, 'YearsSinceLastPromotion': YearsSinceLastPromotion
        }

        st.session_state['input_data'] = input_data

# Function to display prediction results
def display_result():
    input_data = st.session_state.get('input_data', None)

    if input_data:
        # List of feature names (ensure these match the columns used in model training)
        feature_names = ['Age', 'DailyRate', 'Department', 'DistanceFromHome', 'EnvironmentSatisfaction',
                         'Gender', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 
                         'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 
                         'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                         'YearsSinceLastPromotion']

        # Convert the input data to a DataFrame with appropriate feature names
        data_to_predict = pd.DataFrame([[
            input_data['Age'], input_data['DailyRate'], input_data['Department'], input_data['DistanceFromHome'],
            input_data['EnvironmentSatisfaction'], input_data['Gender'], input_data['JobLevel'], input_data['JobRole'],
            input_data['JobSatisfaction'], input_data['MaritalStatus'], input_data['MonthlyIncome'], input_data['MonthlyRate'],
            input_data['NumCompaniesWorked'], input_data['OverTime'], input_data['TotalWorkingYears'],
            input_data['WorkLifeBalance'], input_data['YearsAtCompany'], input_data['YearsInCurrentRole'],
            input_data['YearsSinceLastPromotion']
        ]], columns=feature_names)

        # Use predict_proba to get the probabilities
        probabilities = model.predict_proba(data_to_predict)[0]
        
        # Set a threshold, e.g., 0.5 to classify the employee as likely to leave
        leave_probability = probabilities[1]
        
        # Display the probability and prediction result
        st.header("Final Result :")
        st.write(f"Probability of leaving: {leave_probability * 100:.2f}%")
        
        if leave_probability >= 0.5:  # You can adjust this threshold based on model calibration
            st.error('This employee is likely to **Leave**.')
            st.image('sadimg.png', use_column_width=True)
            st.snow()
            
        else:
            st.success('This employee is likely to **Stay**.')
            st.image('happyimg.png', use_column_width=True)
            st.balloons()
    else:
        st.warning('Please fill out the form first.')


def main():
    if 'input_data' in st.session_state:
        display_result()
    else:
        user_input_form()

# Run the app
if __name__ == '__main__':
    main()
