import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Credit Card Fraud Detection")
st.write("This app uses a SMOTE-trained XGBoost model to predict fraudulent transactions.")
st.sidebar.header("Transaction Features")
st.sidebar.info("Input the V1-V28 PCA components and Amount.")

def user_input_features():
    st.sidebar.subheader("Test Scenarios")
    "Note: V1-V28 are PCA-transformed features from the original credit card dataset to protect user privacy. In a production environment, these would be automated backend signals."
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("Normal Transaction"):
        st.session_state['v_values'] = [0.0] * 28
        st.session_state['amt'] = 50.0
    if col2.button("Suspicious Pattern"):
        st.session_state['v_values'] = [-2.0] * 28 
        st.session_state['amt'] = 999.0
        
    amount = st.sidebar.number_input("Transaction Amount ($)", value=st.session_state.get('amt', 10.0))
    time = st.sidebar.number_input("Seconds Since Last Activity", value=0)
    
    v_dict = {}
    default_vs = st.session_state.get('v_values', [0.0] * 28)
    
    for i in range(1, 29):
        v_dict[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, default_vs[i-1])
    
    data = {'Time': time, 'Amount': amount, **v_dict}
    return pd.DataFrame(data, index=[0])
input_df = user_input_features()
if st.button("Analyze Transaction"):
    temp_df = input_df.copy()
    
    temp_df['amount_scaled'] = scaler.transform(temp_df['Amount'].values.reshape(-1,1))
    temp_df['time_scaled'] = scaler.transform(temp_df['Time'].values.reshape(-1,1))
    
    temp_df.drop(['Time', 'Amount'], axis=1, inplace=True)
    v_cols = [f'V{i}' for i in range(1, 29)]
    expected_order = v_cols + ['amount_scaled', 'time_scaled']
    
    final_input = temp_df[expected_order]
    prediction = model.predict(final_input)
    prob = model.predict_proba(final_input)[0][1]

    st.subheader("Security Analysis Result")
    
    if prediction[0] == 1:
        st.error(f"HIGH RISK: Fraudulent Activity Detected")
        st.metric("Fraud Probability", f"{prob:.2%}", delta="CRITICAL", delta_color="inverse")
        st.warning("Recommended Action: Block Transaction & Notify Customer.")
    else:
        st.success(f"LOW RISK: Genuine Transaction")
        st.metric("Fraud Probability", f"{prob:.2%}", delta="SAFE")
        st.info("Recommended Action: Proceed with Authorization.")

    with st.expander("Technical Details (For Developers)"):
        st.write("V1-V28 are anonymized PCA components representing transaction metadata.")
        st.write("Input Features processed by XGBoost:", final_input)