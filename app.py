import streamlit as st
import pickle
import re
import pdfplumber
from docx import Document
import numpy as np


def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'\b(RT|cc)\b', ' ', cleanText)
    cleanText = re.sub(r'#\S+', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7F]+', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()


# Load the trained models
tfidf = pickle.load(open('tfidfVectorizer.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))

# Category Mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

st.title("Resume Job Prediction System")

st.write("This app predicts the most suitable job categories based on your resume text. Upload your resume file below and click the 'Predict' button.")

uploaded_file = st.file_uploader("Upload your resume (text format only)", type=["txt", "pdf", "docx"])

def is_resume_text(text):
    # List of keywords that are commonly found in resumes
    resume_keywords = ["experience", "education", "skills", "certification", "qualifications", "summary", "work history", "references"]
    # Check if any keyword is present in the text
    return any(keyword in text.lower() for keyword in resume_keywords)

if st.button("Predict"):
    if uploaded_file is not None:
        # Check if the uploaded file is valid
        if uploaded_file.type not in ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            st.error("Invalid file type. Please upload a .txt, .pdf, or .docx file.")
        else:
            # Read the uploaded file content
            raw_data = uploaded_file.read()
            
            # Ensure the file is not empty
            if len(raw_data) == 0:
                st.error("The uploaded file is empty. Please upload a valid resume.")
            else:
                # Attempt to decode the file based on type
                try:
                    if uploaded_file.type == "text/plain":
                        resume_text = raw_data.decode("utf-8")
                    else:
                        # For PDF and DOCX, text extraction approach
                        if uploaded_file.type == "application/pdf":
                            with pdfplumber.open(uploaded_file) as pdf:
                                resume_text = ""
                                for page in pdf.pages:
                                    resume_text += page.extract_text() or ""  # Handle pages with None
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            doc = Document(uploaded_file)
                            resume_text = "\n".join([para.text for para in doc.paragraphs])
                            
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    resume_text = ""
                
                # Check if any text was extracted from the resume
                if not resume_text.strip():
                    st.error("No readable text found in the uploaded file. Please check the content.")
                else:
                    # Check if the text seems to be from a resume
                    if not is_resume_text(resume_text):
                        st.error("The uploaded document does not appear to be a resume. Please upload a valid resume.")
                    else:
                        # Clean the input resume
                        cleaned_resume = cleanResume(resume_text)
                        
                        # Transform the cleaned resume using the trained TfidfVectorizer
                        input_features = tfidf.transform([cleaned_resume])
                        
                        # Make the prediction probabilities using the loaded classifier
                        prediction_probabilities = clf.predict_proba(input_features)[0]
                        
                        # Get the indices of the top 5 predictions
                        top_indices = np.argsort(prediction_probabilities)[-5:][::-1]  # Get indices of top 5 probabilities
                        
                        # Get the category names and probabilities for the top 5 predictions
                        top_categories = [(category_mapping.get(i, "Unknown"), prediction_probabilities[i] * 100) for i in top_indices]

                        # Display the predictions
                        st.success("Top 5 Predicted Job Categories:")
                        for category, probability in top_categories:
                            st.write(f"{category}: {probability:.2f}%")
    else:
        st.error("Please upload a resume file to get a prediction.")
