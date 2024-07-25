import streamlit as st
import joblib
import requests
import pandas as pd
from io import BytesIO

# URLs to the files on GitHub
MODEL_URL = "https://raw.githubusercontent.com/salmakinanthi/prediksi/master/model.pkl"

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

# Define preprocessing functions
def preprocess_data(data):
    # Encoding for categorical columns
    encoder = ce.OrdinalEncoder(cols=['Job Title', 'Location', 'Sector'])  # Adjust based on your model
    data = encoder.fit_transform(data)
    
    # Handle Size and Revenue columns
    data['Size'] = data['Size'].apply(Size)
    data['Revenue'] = data['Revenue'].apply(Revenue)
    
    # Drop unnecessary columns if they were not included in the model
    columns_to_drop = ['Job Description', 'Type of ownership', 'excel', 'spark', 'Company Name', 'Location', 'Founded', 'Competitors', 'Industry', 'hourly', 'employer_provided', 'company_txt', 'job_state', 'python_yn', 'R_yn', 'aws', 'Headquarters']
    
    # Check which columns exist in the DataFrame before attempting to drop them
    columns_existing = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=columns_existing, errors='ignore')
    
    # Return the preprocessed data
    return data

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

def main():
    st.title('Job Salary Prediction App')
    
    st.write("Masukkan data pekerjaan untuk prediksi gaji:")
    
    # Form input for job data
    job_title = st.selectbox('Job Title', ['Software Engineer', 'Data Scientist', 'Product Manager', 'Sales Associate', 'Marketing Manager'])
    location = st.selectbox('Location', ['San Francisco', 'New York', 'Chicago', 'Seattle', 'Austin'])
    company_name = st.text_input('Company Name')  # This may be removed if not used in model
    industry = st.selectbox('Industry', ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing'])  # Ensure this matches
    sector = st.selectbox('Sector', ['Public', 'Private'])
    size = st.selectbox('Size', ['1 to 50 employees', '51 to 200 employees', '201 to 500 employees', '501 to 1000 employees', '1001 to 5000 employees', '5001 to 10000 employees', '10000+ employees'])
    headquarters = st.text_input('Headquarters')  # This may be removed if not used in model
    revenue = st.selectbox('Revenue', ['Unknown / Non-Applicable', '$1 to $2 billion (USD)', '$2 to $5 billion (USD)', '$5 to $10 billion (USD)', '$10+ billion (USD)', '$100 to $500 million (USD)', '$500 million to $1 billion (USD)', '$50 to $100 million (USD)', '$10 to $25 million (USD)', '$25 to $50 million (USD)', '$5 to $10 million (USD)', '$1 to $5 million (USD)'])
    
    submit_button = st.button('Predict')

    if submit_button:
        if model is None:
            st.error("Model is not properly loaded.")
            return
        
        # Create dictionary from input
        data = {
            'Job Title': [job_title],
            'Location': [location],
            'Company Name': [company_name],  # Remove if not used
            'Industry': [industry],  # Remove if not used
            'Sector': [sector],
            'Size': [size],
            'Headquarters': [headquarters],  # Remove if not used
            'Revenue': [revenue]
        }

        # Create DataFrame from the input data
        input_df = pd.DataFrame(data)
        
        # Preprocess data
        input_df = preprocess_data(input_df)

        # Make prediction
        prediction = model.predict(input_df)
        
        st.write(f"Predicted Salary: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
