import streamlit as st
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


pickle_in = open("model_final.pkl","rb")
classifier = pickle.load(pickle_in)

# Function to validate integer input
def is_valid_integer_input(value):
    try:
        int_value = int(value)
        if int_value in (0, 1):
            return True
        else:
            return False
    except ValueError:
        return False

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Disease Score", "Physician Score", "Procedure Score", "Diagnosis Score", "Predict"])

# Main content
if page == "Home":
    st.title("Robin's Claim Prediction system")
    st.write("A friendly web platform to discover claims prediction")

elif page == "Disease Score":
    st.title("Calculate Disease Score")

    # Input fields
    field_names = ['ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke']
    fields = {}
    for field in field_names:
        fields[field] = st.text_input(f"{field} (0 or 1)")

    for field, value in fields.items():
        if value != '':
            fields[field] = int(value)
        else:
            fields[field] = 0

    if st.button("Calculate Disease Score"):
        score = 0
        count = 10
        for field, value in fields.items():
            if value == 1:
                score += count
            count = count-1
        st.success(f"Your Disease Score is: {score}")

    # Convert input values to integers and validate
    #for field, value in fields.items():
    #    if not is_valid_integer_input(value):
    #        st.error(f"Please enter valid integer values (0 or 1) for {field}.")
    #    fields[field] = int(value)




elif page == "Physician Score":
    st.title("Calculate Physician Score")

    # Input fields
    field_names = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']
    fields = {}
    for field in field_names:
        fields[field] = st.text_input(f"{field} (0 or 1)")

    for field, value in fields.items():
        if value != '':
            fields[field] = int(value)
        else:
            fields[field] = 0

    if st.button("Calculate Physician Score"):
        score = 0
        count = 15
        for field, value in fields.items():
            if value == 1:
                score += count
            count = count-5
        st.success(f"Your Physician Score: {score}")



elif page == "Procedure Score":
    st.title("Calculate Procedure Score")

    # Input fields
    field_names = ['ClmProcedureCode_1', 'ClmProcedureCode_2',
       'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
       'ClmProcedureCode_6']
    fields = {}
    for field in field_names:
        fields[field] = st.text_input(f"{field} (0 or 1)")

    for field, value in fields.items():
        if value != '':
            fields[field] = int(value)
        else:
            fields[field] = 0

    if st.button("Calculate Procedure Score"):
        score = 0
        count = 6
        for field, value in fields.items():
            if value == 1:
                score += count
            count = count-1
        st.success(f"Your Procedure Score: {score}")

elif page == "Diagnosis Score":
    st.title("Calculate Diagnosis Score")

    # Input fields
    field_names = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
       'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
       'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
       'ClmDiagnosisCode_10']
    fields = {}
    for field in field_names:
        fields[field] = st.text_input(f"{field} (0 or 1)")

    for field, value in fields.items():
        if value != '':
            fields[field] = int(value)
        else:
            fields[field] = 0

    if st.button("Calculate Diagnosis Score"):
        score = 0
        count = 10
        for field, value in fields.items():
            if value == 1:
                score += count
            count = count-1
        st.success(f"Your Diagnosis Score: {score}")
    

elif page == "Predict":
    st.title("Predict Insurance Amount")

    # Input fields
    field_names = ['BeneID', 'Provider', 'InscClaimAmtReimbursed', 'ClmAdmitDiagnosisCode',
       'DiagnosisGroupCode', 'Days admitted', 'DiagnosisScore',
       'ProcedureScore', 'PhysicianScore', 'NaN count','Total_Score', 'Gender',
       'RenalDiseaseIndicator', 'IPAnnualReimbursementAmt',
       'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt', 'DiseaseScore']
    fields = {}
    for field in field_names:
        fields[field] = st.text_input(f"{field}")

    for field, value in fields.items():
        if value != '':
            fields[field] = int(value)
        else:
            fields[field] = 0


    if st.button('Predict'):
        # Prepare input data
        input_data = []
        for field in field_names:
            if fields[field] != '':
                input_data.append(int(fields[field]))
            else:
                input_data.append(0)
        
        # Convert input data to numpy array for prediction
        input_data = [input_data]  # Convert to list of lists
        input_data = np.array(input_data)  # Convert to numpy array

        # Predict claim amount
        predicted_amount = classifier.predict(input_data)
        
        
        # Display the predicted claim amount
        st.write('Predicted Claim Amount:', predicted_amount)


