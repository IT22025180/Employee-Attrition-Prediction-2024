import streamlit as st

# Title of the app
st.title('Employee Attrition Predictor')

age = st.selectbox(
    'Choose your age:',
    ['18', '19', '20', '21', '22','23','24','25','26','27','28',
     '29','30','31','32','33','34','35','36','37','38',
     '39','40','41','42','43','44','45','46','47','48',
     '49','50','51','52','53','54','55','56','57','58','59','60']
)

dailyRate = st.number_input(
    'Enter your daily rate:',
    min_value=0,      # Minimum value allowed
    key='dailyRate'
)

department = st.selectbox(
    'Choose your department:',
    ['Sales', 'Research & Development', 'Human Resources']
)

distfromhome = st.number_input(
    'Enter distance in KM:',
    min_value=0,      # Minimum value allowed
    max_value=100,    # Maximum value allowed
    step=1,            # Step size
    key='distfromhome'
)

envstatis = st.selectbox(
    'Choose your environment satisfaction:',
    ['4', '3', '2', '1']
)

gender = st.selectbox(
    'Gender:',
    ['Male', 'Female']
)

joblevel = st.selectbox(
    'Choose your job level:',
    ['5', '4', '3', '2', '1']
)

jobrole = st.selectbox(
    'Choose your job role:',
    ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative','Manager','Sales Representative','Research Director','Human Resources']
)

jobstatisfact = st.selectbox(
    'Choose your job satisfaction:',
    ['4', '3', '2', '1']
)

maritalstatus = st.selectbox(
    'Choose your marital status:',
    ['Single', 'Married', 'Divorced']
)

monthlyrate = st.number_input(
    'Enter your monthly rate:',
    min_value=0,      # Minimum value allowed
    key = 'monthlyrate'
)

numOfComp = st.selectbox(
    'Number of companies work with:',
    ['1', '2', '3', '4', '5','6','7','8','9','10']
)

ot = st.selectbox(
    'Choose your over time:',
    ['Yes', 'No']
)

totalworkyrs = st.number_input(
    'Enter distance in KM:',
    min_value=0,      # Minimum value allowed
    max_value=20,    # Maximum value allowed
    step=1,            # Step size
    key='totalworkyrs'
)

workLifeBalance = st.selectbox(
    'Choose your work life balance:',
    ['4', '3', '2', '1']
)

yrsAtcompany = st.number_input(
    'Enter distance in KM:',
    min_value=0,      # Minimum value allowed
    max_value=20,    # Maximum value allowed
    step=1,         # Step size
    key='yrsAtcompany'
)

yearsInCurrentRole = st.selectbox(
    'Choose your years in current role:',
    ['1', '2', '3', '4', '5','6','7','8','9','10',
     '11','12','13','14','15','16','17','18','19','20']
)

yearsSinceLastPromotion = st.selectbox(
    'Choose your year since last promotion:',
    ['0', '1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15']
)

if st.button('View result'):
    st.success('message')

# Display the selected options
# st.write(f'You selected:')
# st.write(f'Category: {category}')
# st.write(f'Size: {size}')
# st.write(f'Color: {color}')

st.balloons()
