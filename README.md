# Resume Screening using TF-IDF and Classification Model

## Overview
This project implements an **AI-powered resume screening system** that predicts the suitability of an uploaded resume for specific job roles. The model utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction and a classification model (clf) to assess the match percentage against predefined job categories.

## Features
- **Resume Upload:** Users can upload their resumes in text format.
- **TF-IDF Vectorization:** Extracts key features from resumes for analysis.
- **Classification Model:** Predicts the resume's suitability for various job roles.
- **Percentage Match:** Provides a percentage-based relevance score.

## Technologies Used
- **Python**
- **scikit-learn** (TF-IDF, classification model)
- **pandas** (data handling)
- **Flask / Streamlit** (optional: for web-based implementation)

## How It Works
1. **Preprocessing:** Extracts text from resumes, removes stop words, and normalizes data.
2. **TF-IDF Transformation:** Converts text data into numerical vectors.
3. **Model Training:** Uses a classification model to learn from labeled job roles.
4. **Prediction:** Compares an uploaded resume against trained categories.
5. **Result Display:** Shows the top job roles with a percentage match.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/vdh05/resume-screening.git
   cd resume-screening
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application (for web interface):
   ```sh
   python app.py
   ```

## Usage
1. Upload a resume file (text format preferred).
2. The model processes the resume and predicts suitable job roles.
3. The system displays a list of job roles with percentage match.

## Example Output
```
Uploaded Resume: Data Scientist Resume.txt

Predicted Job Roles:
- Data Scientist (85%)
- Machine Learning Engineer (78%)
- Software Developer (65%)
```

## Future Enhancements
- **Expand Job Categories:** Include more industries and job roles.
- **Improve Accuracy:** Use deep learning models (e.g., BERT, Transformer-based models).
- **Integrate with ATS:** Connect with Applicant Tracking Systems (ATS) for enterprise use.
- **Web & Mobile App:** Develop a frontend for better user experience.


