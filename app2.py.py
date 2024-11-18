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
# Set the color scheme
primary_color = '#4863A0'
secondary_color = '#4169E1'
tertiary_color = '#368BC1'
background_color = '#F5F5F5'
text_color = '#004225'
font = 'sans serif'
# Set the page config
st.set_page_config(
 page_title='PulmoGenix',
 layout= 'wide',
 initial_sidebar_state='expanded'
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
 background-color: {secondary_color};
 color: {tertiary_color};
 }}
 .streamlit-button {{
 background-color: {primary_color};
 color: {tertiary_color};
 }}
 footer {{
 font-family: {font};
 }}
</style>
""", unsafe_allow_html=True)
# Add university logo to the page
center = st.columns([1])
st.image("uol.jpg", use_column_width=True)
# Molecular descriptor calculator
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
 # Radio button for input method selection
 input_method = st.radio("Choose input method:", ("Copy and Paste SMILES", "Upload 
CSV/TXT File"))
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
 smiles_df = pd.read_csv(uploaded_file, delimiter="\t", header=None, 
names=["SMILES"])
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
 elif st.session_state.page == 'output':
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
if __name__ == "__main__":
 main()
 
# Add a section with the developers' information at the bottom of the page
st.markdown("---")
st.header("PulmoGenix Developers:")
# Add the profiles as individual cards
row1, row2 = st.columns([1, 1])
row3 = st.columns(1)
with row1:
    st.write("")
    st.write("### Dr. Kashif Iqbal Sahibzada")
    #st.write("Assistant Professor")
    st.write("Assistant Professor | Department of Health Professional Technologies, Faculty of Allied Health Sciences, The University of Lahore")
    st.write("Post-Doctoral Fellow | Henan University of Technology,Zhengzhou China ")
    st.write("Email: kashif.iqbal@dhpt.uol.edu.pk | kashif.iqbal@haut.edu.cn")
with row2:
 st.write("")
 st.write("### Munawar Abbas")
 st.write("PhD Scholar")
 st.write("Henan University of Technology,Zhengzhou China")
 st.write("Email: abbas@stu.haut.edu.cn")  
