import streamlit as st
import numpy as np
from urllib.parse import urlparse,parse_qs
import joblib

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
        st.text(Age)

        DailyRate = st.number_input(
            'Enter your daily rate:',
            min_value=0,      # Minimum value allowed
            key='dailyRate'
        )
        st.text(DailyRate)

        Department = st.selectbox(
            'Choose your department:',
            ['2', '1', '0']
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
            ['1', '0']
        )
        st.text(Gender)

        JobLevel = st.selectbox(
            'Choose your job level:',
            ['5', '4', '3', '2', '1']
        )
        st.text(JobLevel)

        JobRole = st.selectbox(
            'Choose your job role:',
            ['0', '1', '2', '3', '4','5','6','7','8']
        )
        st.text(JobRole)

        JobSatisfaction = st.selectbox(
            'Choose your job satisfaction:',
            ['4', '3', '2', '1']
        )
        st.text(JobSatisfaction)

        MaritalStatus = st.selectbox(
            'Choose your marital status:',
            ['2', '1', '0']
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
            ['1', '0']
        )
        st.text(OverTime)

        TotalWorkingYears = st.number_input(
            'Total working years:',
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
        # Convert the inputs to a NumPy array
        data_to_predict = np.array([
            input_data['Age'], input_data['DailyRate'], input_data['Department'], input_data['DistanceFromHome'],
            input_data['EnvironmentSatisfaction'], input_data['Gender'], input_data['JobLevel'], input_data['JobRole'],
            input_data['JobSatisfaction'], input_data['MaritalStatus'], input_data['MonthlyIncome'], input_data['MonthlyRate'],
            input_data['NumCompaniesWorked'], input_data['OverTime'], input_data['TotalWorkingYears'],
            input_data['WorkLifeBalance'], input_data['YearsAtCompany'], input_data['YearsInCurrentRole'],
            input_data['YearsSinceLastPromotion']
        ]).reshape(1, -1)
        
        # Prediction
        prediction = model.predict(data_to_predict)[0]
        
        # Display result
        if prediction == 1:
            st.success('This employee is likely to leave.')
        else:
            st.success('This employee is likely to stay.')
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
