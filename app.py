import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce

# Define functions for feature engineering
def salary(x):
    try:
        a = x.split('-')
        b = int(a[0].replace('$', '').replace('K', '')) + int(a[1].replace('$', '').split("K")[0]) / 2
        return b
    except:
        try:
            a = x.replace('Employer Provided Salary:', '').split('-')
            b = int(a[0].replace('$', '').replace('K', '')) + int(a[1].replace('$', '').split("K")[0]) / 2
        except:
            return ""
        return b

def Size(x):
    if x == '1 to 50 employees':
        return (1 + 50) / 2
    elif x == '51 to 200 employees':
        return (51 + 200) / 2
    elif x == '201 to 500 employees':
        return (201 + 500) / 2
    elif x == '501 to 1000 employees':
        return (501 + 1000) / 2
    elif x == '1001 to 5000 employees':
        return (1001 + 5000) / 2
    elif x == '5001 to 10000 employees':
        return (5001 + 10000) / 2
    elif x == '10000+ employees':
        return 10000
    else:
        return ""

def Revenue(x):
    if x == 'Unknown / Non-Applicable':
        return 0
    elif x == '$1 to $2 billion (USD)':
        return (1 + 2) / 2
    elif x == '$2 to $5 billion (USD)':
        return (2 + 5) / 2
    elif x == '$5 to $10 billion (USD)':
        return (5 + 10) / 2
    elif x == '$10+ billion (USD)':
        return 10
    elif x == '$100 to $500 million (USD)':
        return (100 + 500) / 2
    elif x == '$500 million to $1 billion (USD)':
        return (500 + 1000) / 2
    elif x == '$50 to $100 million (USD)':
        return (50 + 100) / 2
    elif x == '$10 to $25 million (USD)':
        return (10 + 25) / 2
    elif x == '$25 to $50 million (USD)':
        return (25 + 50) / 2
    elif x == '$5 to $10 million (USD)':
        return (5 + 10) / 2
    elif x == '$1 to $5 million (USD)':
        return (1 + 5) / 2
    else:
        return ""

# Load the pre-trained model
@st.cache
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Streamlit UI
st.title("Salary Prediction Model")

# Input fields for user data
st.sidebar.header("Input Data")
input_data = {
    'Job Title': st.sidebar.text_input('Job Title', value=""),
    'Location': st.sidebar.text_input('Location', value=""),
    'Company Name': st.sidebar.text_input('Company Name', value=""),
    'Industry': st.sidebar.text_input('Industry', value=""),
    'Sector': st.sidebar.text_input('Sector', value=""),
    'Size': st.sidebar.text_input('Size', value=""),
    'Headquarters': st.sidebar.text_input('Headquarters', value=""),
    'Revenue': st.sidebar.text_input('Revenue', value="")
}

# Create a button for prediction
if st.sidebar.button('Predict'):
    # Create DataFrame for input data
    input_df = pd.DataFrame([input_data])

    # Data processing for input data
    input_df['Size'] = input_df['Size'].apply(lambda x: Size(x))
    input_df['Revenue'] = input_df['Revenue'].apply(lambda x: Revenue(x))

    encoder = ce.OrdinalEncoder(cols=['Job Title', 'Location', 'Company Name', 'Industry', 'Sector', 'Headquarters'])
    input_df = encoder.fit_transform(input_df)

    # Make prediction
    prediction = model.predict(input_df)
    st.write(f"Predicted Salary: ${prediction[0]:,.2f}")
