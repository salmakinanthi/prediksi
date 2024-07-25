import streamlit as st
import joblib
import requests
import pandas as pd
from io import BytesIO

# URLs to the files on GitHub
MODEL_URL = "https://raw.githubusercontent.com/salmakinanthi/uasmpml/master/best_model.pkl"

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
        model, feature_names = joblib.load(model_file)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
else:
    model = None



# Function to convert categorical data to numerical data using Ordinal Encoding
def encode_data(data):
    import category_encoders as ce
    encoder = ce.OrdinalEncoder(cols=['Job Title', 'Location', 'Company Name', 'Industry', 'Sector', 'Headquarters'])
    data = encoder.fit_transform(data)
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
        return np.nan

# Function to convert revenue to numerical values
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
        return np.nan

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

# Encode the input data
input_data_encoded = encode_data(input_data)

# Make predictions
if st.button('Predict Salary'):
    prediction = model.predict(input_data_encoded)
    st.write("The estimated salary is: ${:.2f}K".format(prediction[0]))
