import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from resume_classification import predict_category
import io
import codecs
import pdfminer.high_level

# Download the NLTK stopwords if they haven't been downloaded yet
nltk.download('stopwords')

# Set up the Streamlit app
st.title('Resume Classifier with Score')

# Add a text box for the user to enter their reference text
ref_text = st.text_area('Enter your requirements')

# Add a file uploader for the user to upload multiple files
uploaded_files = st.file_uploader('Upload files', accept_multiple_files=True)

# Define a function to calculate the similarity score between the reference text and each uploaded file
def calculate_similarity(ref_text, file_contents):
    # Create a TfidfVectorizer object to convert text to vectors
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    # Convert the reference text and file contents to vectors
    ref_vector = vectorizer.fit_transform([ref_text])
    file_vector = vectorizer.transform([file_contents])

    # Calculate the cosine similarity score between the two vectors
    similarity_score = cosine_similarity(ref_vector, file_vector)[0][0]

    return similarity_score

# Define a function to process the uploaded files and sort them by similarity score

def process_files(ref_text, uploaded_files):
    # Create lists to store the file names, similarity scores and resume class
    name = []
    score = []
    resume_class = []
    # Loop through each uploaded file
    for file in uploaded_files:
        # Read the file contents with the correct encoding
        file_contents = io.BytesIO(file.read())

        # Check if the file is a PDF
        if file.name.endswith('.pdf'):
            # Use pdfminer to extract the text from the PDF
            extracted_text = pdfminer.high_level.extract_text(file_contents)
        else:
            # Read the file contents as text
            extracted_text = file_contents.read().decode('utf-8')

        # Calculate the similarity score between the reference text and the file contents
        similarity_score = calculate_similarity(ref_text, extracted_text)

        file_name = file.name
        name.append(file_name)
        score.append(similarity_score)
        resume_cat = predict_category(extracted_text)
        resume_class.append(resume_cat)
        # Append the filename and similarity score to the DataFrame
        results_df = pd.DataFrame(list(zip(name, score, resume_class)), 
        columns=["Filename", "Score", "Resume class"])
        output_df1 = results_df.astype(str).replace(
                    {"\[": "", "\]": "", "\'": ""}, regex=True).astype(str)
    # Sort the DataFrame by similarity score in descending order
    results_df = output_df1.sort_values('Score', ascending=False)

    return results_df



# Add a button to submit the reference text and uploaded files
if st.button('Submit'):
    # Check if the user has entered any reference text
    if ref_text == '':
        st.warning('Please enter your requirements text')
    # Check if the user has uploaded any files
    elif uploaded_files is None:
        st.warning('Please upload some files')
    else:
        # Process the uploaded files and display the results
        results_df = process_files(ref_text, uploaded_files)
        st.write(results_df)

# reset_button = st.button('Reset')
# Add a button to reset the uploaded files
if st.button('Reset'):
    uploaded_files = None