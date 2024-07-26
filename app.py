import streamlit as st
import joblib
import requests
import pandas as pd
from io import BytesIO
import numpy as np

# URLs to the files on GitHub
MODEL_URL = "https://raw.githubusercontent.com/salmakinanthi/prediksi/main/model.pkl"

def load_file_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file from {url}: {e}")
        return None

# Load the model directly from URL
model_file = load_file_from_url(MODEL_URL)

if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        model = None
else:
    model = None

# Manual encoding mappings
job_title_mapping = {
    'Data Scientist': 1,
    'Data Engineer': 2,
    'Principal Data Scientist with over 10 years experience': 3,
    'Data Operations Lead': 4,
    'Research Scientist, Immunology - Cancer Biology': 5,
    'Senior Scientist, Cell Pharmacology/Assay Development': 6,
    'Scientist – …': 7
}

headquarters_mapping = {
    'San Francisco, CA': 1,
    'New York, NY': 2,
    'Chicago, IL': 3,
    'Austin, TX': 4,
    'Seattle, WA': 5
}

sector_mapping = {
    'Information Technology': 1,
    'Finance': 2,
    'Healthcare': 3,
    'Consumer Goods': 4
}

# Function to encode data manually
def encode_data(data):
    data['Job Title'] = data['Job Title'].map(job_title_mapping).fillna(-1)
    data['Headquarters'] = data['Headquarters'].map(headquarters_mapping).fillna(-1)
    data['Sector'] = data['Sector'].map(sector_mapping).fillna(-1)
    return data

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

# Custom CSS for pastel theme
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
        }
        .stApp {
            background-color: #f7f7f7;
            color: #333;
        }
        .stButton button {
            background-color: #ffb6c1;
            border-radius: 12px;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            transition-duration: 0.4s;
        }
        .stButton button:hover {
            background-color: #ff6f61;
            color: white;
        }
        .stSelectbox, .stNumberInput, .stTextInput {
            border: 1px solid #ffb6c1;
            border-radius: 4px;
            padding: 8px;
        }
        .stMarkdown, .stText {
            color: #555;
        }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Salary Prediction App", page_icon=":moneybag:", layout="wide")

st.title("Salary Prediction App")

with st.sidebar:
    st.header("User Input Features")
    st.markdown("Please provide the following details:")

    job_title = st.selectbox("Job Title", list(job_title_mapping.keys()))
    age = st.number_input('Age', min_value=0, max_value=100, value=25)
    headquarters = st.selectbox("Headquarters", list(headquarters_mapping.keys()))
    sector = st.selectbox("Sector", list(sector_mapping.keys()))
    size = st.selectbox("Company Size", [
        '1 to 50 employees', '51 to 200 employees', '201 to 500 employees', 
        '501 to 1000 employees', '1001 to 5000 employees', '5001 to 10000 employees', 
        '10000+ employees'])
    revenue = st.selectbox("Company Revenue", [
        'Unknown / Non-Applicable', '$1 to $2 billion (USD)', '$2 to $5 billion (USD)', 
        '$5 to $10 billion (USD)', '$10+ billion (USD)', '$100 to $500 million (USD)', 
        '$500 million to $1 billion (USD)', '$50 to $100 million (USD)', '$10 to $25 million (USD)', 
        '$25 to $50 million (USD)', '$5 to $10 million (USD)', '$1 to $5 million (USD)'])

input_data = pd.DataFrame({
    'Job Title': [job_title],
    'Headquarters': [headquarters],
    'Size': [Size(size)],
    'Sector': [sector],
    'Revenue': [Revenue(revenue)],
    'age': [age]
})

# Check for NaNs in the input data
if input_data.isnull().values.any():
    st.error("Please fill out all fields correctly.")
else:
    # Encode the input data manually
    input_data_encoded = encode_data(input_data)

    # Make predictions
    if model is not None:
        if st.button('Predict Salary'):
            if input_data_encoded is not None:
                try:
                    prediction = model.predict(input_data_encoded)
                    st.success(f"The estimated salary is: ${prediction[0]:.2f}K")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Error encoding input data.")
    else:
        st.warning("Model not loaded or not ready.")
