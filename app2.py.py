# Importing necessary libraries
import streamlit as st
import requests
import pandas as pd
import numpy as np
from PIL import Image
import subprocess
import os
import base64
import pickle
import joblib
from joblib import dump, load
import sklearn
from sklearn import svm
from sklearn import datasets
# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title='PharmacoGenix',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='💊',
)

def main():
    # Set the color scheme
    header_color = '#800000'         # Maroon
    background_color = '#FFFFFF'     # White
    text_color = '#333333'           # Dark Gray
    primary_color = '#A52A2A'        # Darker Maroon
    footer_color = '#550000'         # Deep Maroon
    footer_text_color = '#FFFFFF'    # White
    font = 'Arial, sans-serif'

    # Set the theme
    st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
            color: {text_color};
            font-family: {font};
        }}
        .sidebar .sidebar-content {{
            background-color: {header_color};
            color: {text_color};
        }}
        .stButton > button {{
            background-color: {primary_color};
            color: {background_color};
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 20px;
        }}
        footer {{
            font-family: {font};
            background-color: {footer_color};
            color: {footer_text_color};
        }}
        .header-title {{
            color: {primary_color};
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }}
        .header-subtitle {{
            color: {text_color};
            font-size: 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
    </style>
    """, unsafe_allow_html=True)

   # Add header with application title and description
with st.container():  # Corrected from 'center' to 'st.container'
    st.markdown(
        "<h1 class='header-title'>PharmacoGenix – An Artificial Intelligence Approach towards the Drug Discovery</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p class='header-subtitle'>
        Welcome to PharmacoGenix, a powerful prediction server designed to assess the pIC50 values of compounds targeting Ribosome Methyltransferase (erm 41). Built on a highly accurate machine learning-based regression model, PharmacoGenix achieves an impressive 99% accuracy, enabling precise and reliable predictions. This tool deciphers complex molecular interactions, providing insights into the inhibitory potential of phytochemicals, microbial peptides, archaeal peptides, and synthetic ligands. Join us in advancing antimicrobial research, unlocking novel therapeutic possibilities against ribosomal resistance mechanisms.
        </p>
        """,
        unsafe_allow_html=True
    )
    #st.image("erm.jpg", width=800)
    col1, col2, col3 = st.columns([1,2,3])
    with col2:
        st.image("erm.jpg", width=800)

if __name__ == "__main__":
    main()
def main():
    # Initialize session state variables if not already set
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = ''
    if 'prediction_df' not in st.session_state:
        st.session_state.prediction_df = None
    # Navigation function
    def navigate_to(page):
        st.session_state.page = page

    # Input page
    if st.session_state.page == 'input':
        st.subheader('pIC50 Prediction')

        # Radio button for input method selection (Unique Key Added)
        input_method = st.radio(
            "Choose input method:", 
            ("Copy and Paste SMILES", "Upload CSV/TXT File"), 
            key="input_method_radio"
        )

        if input_method == "Copy and Paste SMILES":
            st.header('1. Enter SMILES String:')
            smiles_input = st.text_area("Enter SMILES String here:", "", key="smiles_text_area")

            if st.button('Predict', key="predict_button_text"):
                if smiles_input:
                    # Store SMILES input in session state
                    st.session_state.smiles_input = smiles_input

                    # Perform prediction and store results
                    st.session_state.prediction_df = handle_prediction(smiles_input)

                    # Navigate to the output page
                    navigate_to('output')
                else:
                    st.warning('Please enter a SMILES string.')

        else:
            st.header('1. Upload CSV or TXT file containing SMILES:')
            uploaded_file = st.file_uploader("Upload file", type=["csv", "txt"], key="file_uploader")

            if st.button('Predict', key="predict_button_file"):
                if uploaded_file is not None:
                    # Read the uploaded file
                    smiles_df = pd.read_csv(uploaded_file)
                    smiles_input = '\n'.join(smiles_df.iloc[:, 0].astype(str))

                    # Store SMILES input in session state
                    st.session_state.smiles_input = smiles_input

                    # Perform prediction and store results
                    st.session_state.prediction_df = handle_prediction(smiles_input)

                    # Navigate to the output page
                    navigate_to('output')
                else:
                    st.warning('Please upload a valid file.')

    # Output page
    elif st.session_state.page == 'output':
        st.subheader('Prediction Results')

        if st.session_state.prediction_df is not None:
            st.dataframe(st.session_state.prediction_df)

            # Download option
            st.markdown(filedownload(st.session_state.prediction_df), unsafe_allow_html=True)

            if st.button('Go Back', key="go_back_button"):
                navigate_to('input')
        else:
            st.error("No prediction data available. Please go back and provide input.")

# Descriptor Calculation Function
def desc_calc(smiles_input):
    try:
        bashCommand = f"your_command_here {smiles_input}"  # Ensure this is correct
        process = subprocess.run(bashCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        
        if process.returncode != 0:
            st.error(f"Error in descriptor calculation: {process.stderr}")
            return None
        
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"Subprocess error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None
# File download function
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model prediction function
def build_model(input_data, smiles_list):
    load_model = pickle.load(open('model.pkl', 'rb'))
    prediction = load_model.predict(input_data)
    df = pd.DataFrame({
        'Canonical Smiles': smiles_list,
        'pIC50 values': prediction
    })
    df_sorted = df.sort_values(by='pIC50 values', ascending=False).reset_index(drop=True)
    df_sorted.index += 1
    return df_sorted

# Function to handle SMILES input and prediction
def handle_prediction(smiles_input):
    desc_result = desc_calc(smiles_input)
    if desc_result is None:
        return None
    
    try:
        prediction_df = pd.DataFrame({'SMILES': [smiles_input], 'Prediction': [desc_result]})
        return prediction_df
    except Exception as e:
        st.error(f"Error in processing prediction: {str(e)}")
        return None

if __name__ == "__main__":
    main()

 
# HTML and CSS to color the title and header
st.markdown(
    """
    <style>
    .title {
        color: #800000;  /* Parrot Green color code */
        font-size: 2em;
        font-weight: bold;
    }
    .header {
        color: #800000;  /* Parrot Green color code */
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
    <h1 class="title">Team PharmacoGenix:</h1>
    """,
    unsafe_allow_html=True
)
 
# Define columns for the profiles
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # st.image("my-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Dr. Kashif Iqbal Sahibzada</h3>
             Assistant Professor | Department of Health Professional Technologies, Faculty of Allied Health Sciences, The University of Lahore<br>
            Post-Doctoral Fellow | Henan University of Technology,Zhengzhou China<br>
            Email: kashif.iqbal@dhpt.uol.edu.pk | kashif.iqbal@haut.edu.cn
        </div>
    """, unsafe_allow_html=True)

with col2:
    # st.image("colleague-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Munawar Abbas</h3>
            PhD Scholar<br>
            Henan University of Technology,Zhengzhou China<br>
            Email: abbas@stu.haut.edu.cn
        </div>
    """, unsafe_allow_html=True)

with col3:
    # st.image("teacher-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Shumaila Shahid</h3>
            MS Biochemistry<br>
            School of Biochemistry and Biotechnology<br>
            University of the Punjab, Lahore<br>
            Email: shumaila.ms.sbb@pu.edu.pk
        </div>
    """, unsafe_allow_html=True)

#Add University Logo
left_logo, center_left, center_right, right_logo = st.columns([1, 1, 1, 1])
#left_logo.image("LOGO_u.jpeg", width=200)
center_left.image("uol.jpg", width=450)  # Replace with your center-left logo image
#right_logo.image("image.jpg", width=200)
