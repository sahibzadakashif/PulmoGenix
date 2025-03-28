#Importing necessary libraries
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
def main():
    # Set the color scheme
    header_color = '#91C788'
    background_color = '#FFFFFF'
    text_color = '#333333'
    primary_color = '#800000'
    footer_color = '#017C8C'
    footer_text_color = '#FFFFFF'
    font = 'Arial, sans serif'

    # Set the page config
    st.set_page_config(
        page_title='PharmacoGenix',
        layout='wide',
        initial_sidebar_state='expanded',
        page_icon='💊',
    )

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
     # Add the image and title at the top of the page
    col1, col2, col3 = st.columns([1,2,3])
    with col1:
        st.image("erm.jpg", width=580)
    with col3:
        st.markdown("<h1 class='header-title'>PharmacoGenix – An AI-Based Approach towards Drug Discovery</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p class='header-subtitle'>
        Welcome to PharmacoGenix, a powerful prediction server designed to assess the pIC50 values of compounds targeting Ribosome Methyltransferase (erm 41). Built on a highly accurate machine learning-based regression model, PharmacoGenix achieves an impressive 99% accuracy, enabling precise and reliable predictions. This tool deciphers complex molecular interactions, providing insights into the inhibitory potential of phytochemicals, microbial peptides, archaeal peptides, and synthetic ligands. Join us in advancing antimicrobial research, unlocking novel therapeutic possibilities against ribosomal resistance mechanisms.
        </p>
        """, unsafe_allow_html=True)
# Add university logos to the page
    left_logo, center, right_logo = st.columns([1, 2, 1])
    #center.image("ref.jpg", width=650)
    #right_logo.image("image.jpg", width=250)
if __name__ == "__main__":
    main()
# Molecular descriptor calculator
def desc_calc(smiles_input):
    # Writes SMILES input to a file
    with open('molecule(1).smi', 'w') as f:
        f.write(smiles_input)
    
    # Corrected and properly terminated bash command
    bashCommand = (
        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
        "-removesalt -standardizenitro -fingerprints "
        "-descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml "
        "-dir ./ -file descriptors_output.csv"
    )
    
    # Perform the descriptor calculation
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Remove the temporary SMILES input file
    os.remove('molecule(1).smi')
# File download
def filedownload(df):
 csv = df.to_csv(index=False)
 b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
 href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'

 return href
def build_model(input_data, smiles_list):
 # Reads in saved regression model
 load_model = pickle.load(open('model.pkl', 'rb'))
 # Apply model to make predictions
 prediction = load_model.predict(input_data)
 # Create DataFrame for predictions
 prediction_output = pd.Series(prediction, name='pIC50 values')
 # Create DataFrame with predictions
 df = pd.DataFrame({
 'Canonical Smiles': smiles_list,
 'pIC50 values': prediction_output
 })
# Sort DataFrame by pIC50 values in descending order and reset index
 df_sorted = df.sort_values(by='pIC50 values', ascending=False).reset_index(drop=True)
 df_sorted.index += 1 # Start index from 1
 return df_sorted
# Function to handle SMILES input and prediction
def handle_prediction(smiles_input):
    with st.spinner("Calculating descriptors..."):
        desc_calc(smiles_input)
    
    # Read in calculated descriptors
    desc = pd.read_csv('descriptors_output.csv')
    
    # Read descriptor list used in previously built model
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    
    # Split SMILES strings
    smiles_list = smiles_input.split('\n')
    
    # Get prediction results
    prediction_df = build_model(desc_subset, smiles_list)
    return prediction_df
 # Main function to handle the app's logic
# Main function to handle the app's logic
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'input'

    # Navigation function
    def navigate_to(page):
        st.session_state.page = page

    # Input page
    if st.session_state.page == 'input':
        st.subheader('pIC50 Prediction')
        # Further code for the input page here...
# Radio button for input method selection
input_method = st.radio("Choose input method:", ("Copy and Paste SMILES", "Upload CSV/TXT File"))

if input_method == "Copy and Paste SMILES":
    st.header('1. Enter SMILES String:')
    smiles_input = st.text_area("Enter SMILES String here:", "")
    
    if st.button('Predict'):
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
    uploaded_file = st.file_uploader("Upload file", type=["csv", "txt"])
    
    if st.button('Predict'):
        if uploaded_file is not None:
            # Read the uploaded file
            if uploaded_file.type == "text/csv":
                smiles_df = pd.read_csv(uploaded_file)
            else:
                smiles_df = pd.read_csv(uploaded_file, delimiter="\t", header=None, names=["SMILES"])
            
            smiles_list = smiles_df['SMILES'].tolist()
            smiles_input = '\n'.join(smiles_list)
            
            # Store SMILES input in session state
            st.session_state.smiles_input = smiles_input
            
            # Perform prediction and store results
            st.session_state.prediction_df = handle_prediction(smiles_input)
            
            # Navigate to the output page
            navigate_to('output')
        else:
            st.warning('Please upload a CSV or TXT file.')

# Output page
if st.session_state.page == 'output':
    st.header('**Prediction output**')
    
    # Center the prediction table
    st.markdown(
        """
        <style>
        .center-table {
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    st.table(st.session_state.prediction_df.style.set_table_styles(
        [{'selector': '', 'props': [('margin-left', 'auto'), ('margin-right', 'auto')]}]
    ))
    
    st.markdown(filedownload(st.session_state.prediction_df), unsafe_allow_html=True)
    
    if st.button('Go Back'):
        navigate_to('input')
 
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
    <h1 class="title">Team OctaScanner:</h1>
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
