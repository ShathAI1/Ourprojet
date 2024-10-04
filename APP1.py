import streamlit as st
import numpy as np
import joblib
import pickle as p
from sklearn.preprocessing import StandardScaler

# Load the models and scaler
with open('Ourfinalisedmodel.pickle', 'rb') as f:
    pipe = p.load(f)

rf_model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

def main():
    st.title('Energy Prediction App')

    # Input fields
    purchases = st.number_input('Purchases', value=0.0)
    bTotal = st.number_input('bTotal', value=0.0)
    # Other inputs...

    if st.button('Predict'):
        features = np.array([[purchases, bTotal, ...]])  # fill in other inputs

        # Use scaler or pipe based on your use case
        features_scaled = scaler.transform(features)
        prediction = rf_model.predict(features_scaled)[0]

        st.success(f'The predicted value is: {prediction}')

if __name__ == '__main__':
    main()
