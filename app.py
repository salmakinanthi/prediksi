import streamlit as st
import joblib
import requests
import pandas as pd
from io import BytesIO
import numpy as np

# URLs to the files on GitHub
MODEL_URL = "https://raw.githubusercontent.com/salmakinanthi/prediksi/main/lrmodel.pkl"
ENCODER_URL = "https://raw.githubusercontent.com/salmakinanthi/prediksi/main/encoder.pkl"

def load_file_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file from {url}: {e}")
        return None

# Load the model and encoder directly from URL
model_file = load_file_from_url(MODEL_URL)
encoder_file = load_file_from_url(ENCODER_URL)

if model_file is not None and encoder_file is not None:
    try:
        model = joblib.load(model_file)
        encoder = joblib.load(encoder_file)
        st.write("Model and encoder loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the model or encoder: {e}")
        model = encoder = None
else:
    model = encoder = None

# Function to convert categorical data to numerical data using Ordinal Encoding
def encode_data(data):
    if encoder is not None:
        data = encoder.transform(data)
    return data

# Function to parse salary range and compute average salary
def salary(x):
    try:
        a = x.split('-')
        b = (int(a[0].replace('$', '').replace('K', '')) + int(a[1].replace('$', '').replace('K', ''))) / 2
        return b
    except:
        try:
            a = x.replace('Employer Provided Salary:', '').split('-')
            b = (int(a[0].replace('$', '').replace('K', '')) + int(a[1].replace('$', '').replace('K', ''))) / 2
        except:
            return np.nan
        return b

# Function to convert company size to numerical values
def Size(x):
    size_mapping = {
        '1 to 50 employees': (1 + 50) / 2,
        '51 to 200 employees': (51 + 200) / 2,
        '201 to 500 employees': (201 + 500) / 2,
        '501 to 1000 employees': (501 + 1000) / 2,
        '1001 to 5000 employees': (1001 + 5000) / 2,
        '5001 to 10000 employees': (5001 + 10000) / 2,
        '10000+ employees': 10000
    }
    return size_mapping.get(x, np.nan)

# Function to convert revenue to numerical values
def Revenue(x):
    revenue_mapping = {
        'Unknown / Non-Applicable': 0,
        '$1 to $2 billion (USD)': (1 + 2) / 2,
        '$2 to $5 billion (USD)': (2 + 5) / 2,
        '$5 to $10 billion (USD)': (5 + 10) / 2,
        '$10+ billion (USD)': 10,
        '$100 to $500 million (USD)': (100 + 500) / 2,
        '$500 million to $1 billion (USD)': (500 + 1000) / 2,
        '$50 to $100 million (USD)': (50 + 100) / 2,
        '$10 to $25 million (USD)': (10 + 25) / 2,
        '$25 to $50 million (USD)': (25 + 50) / 2,
        '$5 to $10 million (USD)': (5 + 10) / 2,
        '$1 to $5 million (USD)': (1 + 5) / 2
    }
    return revenue_mapping.get(x, np.nan)

# Streamlit app
st.title("Salary Prediction App")

# Input features from user
job_title = st.text_input("Job Title")
location = st.text_input("Location")
company_name = st.text_input("Company Name")
industry = st.text_input("Industry")
sector = st.text_input("Sector")
headquarters = st.text_input("Headquarters")
size = st.selectbox("Company Size", ['1 to 50 employees', '51 to 200 employees', '201 to 500 employees', '501 to 1000 employees', '1001 to 5000 employees', '5001 to 10000 employees', '10000+ employees'])
revenue = st.selectbox("Company Revenue", ['Unknown / Non-Applicable', '$1 to $2 billion (USD)', '$2 to $5 billion (USD)', '$5 to $10 billion (USD)', '$10+ billion (USD)', '$100 to $500 million (USD)', '$500 million to $1 billion (USD)', '$50 to $100 million (USD)', '$10 to $25 million (USD)', '$25 to $50 million (USD)', '$5 to $10 million (USD)', '$1 to $5 million (USD)'])

# Create a dataframe for the input data
input_data = pd.DataFrame({
    'Job Title': [job_title],
    'Location': [location],
    'Company Name': [company_name],
    'Industry': [industry],
    'Sector': [sector],
    'Headquarters': [headquarters],
    'Size': [Size(size)],
    'Revenue': [Revenue(revenue)]
})

# Check for NaNs in the input data
if input_data.isnull().values.any():
    st.error("Please fill out all fields correctly.")
else:
    # Encode the input data
    input_data_encoded = encode_data(input_data)

    # Make predictions
    if model is not None and st.button('Predict Salary'):
        try:
            prediction = model.predict(input_data_encoded)
            st.write("The estimated salary is: ${:.2f}K".format(prediction[0]))
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model not loaded or not ready.")
