import streamlit as st
import numpy as np
from urllib.parse import urlparse,parse_qs

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
    Age = st.selectbox(
        'Choose your age:',
        ['18', '19', '20', '21', '22','23','24','25','26','27','28',
        '29','30','31','32','33','34','35','36','37','38',
        '39','40','41','42','43','44','45','46','47','48',
        '49','50','51','52','53','54','55','56','57','58','59','60']
    )
    st.text(Age)

    DailyRate = st.number_input(
        'Enter your daily rate:',
        min_value=0,      # Minimum value allowed
        key='dailyRate'
    )
    st.text(DailyRate)

    Department = st.selectbox(
        'Choose your department:',
        ['Sales', 'Research & Development', 'Human Resources']
    )
    st.text(Department)

    DistanceFromHome = st.number_input(
        'Enter distance in KM:',
        min_value=0,      # Minimum value allowed
        max_value=100,    # Maximum value allowed
        step=1,            # Step size
        key='distfromhome'
    )
    st.text(DistanceFromHome)

    EnvironmentSatisfaction = st.selectbox(
        'Choose your environment satisfaction:',
        ['4', '3', '2', '1']
    )
    st.text(EnvironmentSatisfaction)

    Gender = st.selectbox(
        'Gender:',
        ['Male', 'Female']
    )
    st.text(Gender)

    JobLevel = st.selectbox(
        'Choose your job level:',
        ['5', '4', '3', '2', '1']
    )
    st.text(JobLevel)

    JobRole = st.selectbox(
        'Choose your job role:',
        ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative','Manager','Sales Representative','Research Director','Human Resources']
    )
    st.text(JobRole)

    JobSatisfaction = st.selectbox(
        'Choose your job satisfaction:',
        ['4', '3', '2', '1']
    )
    st.text(JobSatisfaction)

    MaritalStatus = st.selectbox(
        'Choose your marital status:',
        ['Single', 'Married', 'Divorced']
    )
    st.text(MaritalStatus)

    MonthlyIncome = st.number_input(
        'Enter your monthly income:',
        min_value=0,      # Minimum value allowed
        key = 'monthlyincome'
    )
    st.text(MonthlyIncome)

    MonthlyRate = st.number_input(
        'Enter your monthly rate:',
        min_value=0,      # Minimum value allowed
        key = 'monthlyrate'
    )
    st.text(MonthlyRate)

    NumCompaniesWorked = st.selectbox(
        'Number of companies work with:',
        ['1', '2', '3', '4', '5','6','7','8','9','10']
    )
    st.text(NumCompaniesWorked)

    OverTime = st.selectbox(
        'Choose your over time:',
        ['Yes', 'No']
    )
    st.text(OverTime)

    TotalWorkingYears = st.number_input(
        'Enter distance in KM:',
        min_value=0,      # Minimum value allowed
        max_value=20,    # Maximum value allowed
        step=1,            # Step size
        key='totalworkyrs'
    )
    st.text(TotalWorkingYears)

    WorkLifeBalance = st.selectbox(
        'Choose your work life balance:',
        ['4', '3', '2', '1']
    )
    st.text(WorkLifeBalance)

    YearsAtCompany = st.number_input(
        'Years at company:',
        min_value=0,      # Minimum value allowed
        max_value=20,    # Maximum value allowed
        step=1,         # Step size
        key='yrsAtcompany'
    )
    st.text(YearsAtCompany)

    YearsInCurrentRole = st.selectbox(
        'Choose your years in current role:',
        ['1', '2', '3', '4', '5','6','7','8','9','10',
        '11','12','13','14','15','16','17','18','19','20']
    )
    st.text(YearsInCurrentRole)

    YearsSinceLastPromotion = st.selectbox(
        'Choose your year since last promotion:',
        ['0', '1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15']
    )
    st.text(YearsSinceLastPromotion)

    inputs = {
        'Age': Age, 'DailyRate': DailyRate, 'Department': Department, 'DistanceFromHome': DistanceFromHome,
        'EnvironmentSatisfaction': EnvironmentSatisfaction, 'Gender': Gender, 'JobLevel': JobLevel, 'JobRole': JobRole,
        'JobSatisfaction': JobSatisfaction, 'MaritalStatus': MaritalStatus,'MonthlyIncome': MonthlyIncome, 'MonthlyRate': MonthlyRate,
        'NumCompaniesWorked': NumCompaniesWorked, 'OverTime': OverTime, 'TotalWorkingYears': TotalWorkingYears, 'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany, 'YearsInCurrentRole': YearsInCurrentRole, 'YearsSinceLastPromotion': YearsSinceLastPromotion
    }

    return inputs

display_page()
