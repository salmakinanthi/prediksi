import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce

# Load the pre-trained model
@st.cache
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Define preprocessing function
def preprocess_data(data):
    # Encoding for categorical columns
    encoder = ce.OrdinalEncoder(cols=['Job Title', 'Location', 'Company Name', 'Industry', 'Sector', 'Headquarters'])
    data = encoder.fit_transform(data)
    
    # Handle Size and Revenue columns
    data['Size'] = data['Size'].apply(lambda x: Size(x))
    data['Revenue'] = data['Revenue'].apply(lambda x: Revenue(x))
    
    # Drop unnecessary columns if they were not included in the model
    columns_to_drop = ['Job Description', 'Type of ownership', 'excel', 'spark', 'Company Name', 'Location', 'Founded', 'Competitors', 'Industry', 'hourly', 'employer_provided', 'company_txt', 'job_state', 'python_yn', 'R_yn', 'aws']
    
    # Check which columns exist in the DataFrame before attempting to drop them
    columns_existing = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=columns_existing)
    
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
    company_name = st.text_input('Company Name')
    industry = st.selectbox('Industry', ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing'])
    sector = st.selectbox('Sector', ['Public', 'Private'])
    size = st.selectbox('Size', ['1 to 50 employees', '51 to 200 employees', '201 to 500 employees', '501 to 1000 employees', '1001 to 5000 employees', '5001 to 10000 employees', '10000+ employees'])
    headquarters = st.text_input('Headquarters')
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
            'Company Name': [company_name],
            'Industry': [industry],
            'Sector': [sector],
            'Size': [size],
            'Headquarters': [headquarters],
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
