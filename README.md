# AI Resume Analyzer 📄🤖

An AI-powered resume analysis project designed to help freshers review their resumes and identify areas for improvement.

## Overview

Creating a strong resume can be challenging when starting a career. This project provides AI-assisted feedback to help users improve resume clarity, highlight relevant skills, and present their qualifications more effectively.

## Key Features

- AI-assisted resume analysis
- An indicative ATS score
- Feedback on resume content and presentation
- Suggestions for improving skills and project descriptions
- A fresher-focused approach without requiring work experience

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- Git and GitHub

## Run Locally

### 1. Clone the repository

    git clone https://github.com/42111392-vendhan/ai-resume-analyzer.git
    cd ai-resume-analyzer

### 2. Create a virtual environment

    python -m venv venv

### 3. Activate the environment

Windows Command Prompt:

    venv\Scripts\activate

Linux / macOS:

    source venv/bin/activate

### 4. Install dependencies

    pip install -r requirements.txt

### 5. Configure your API key

Configure your Google Gemini API key using the secret or environment-variable name expected by `app.py`.

Never commit API keys or local secrets to GitHub.

### 6. Start the application

    streamlit run app.py

## How to Use

1. Open the application in your browser.
2. Provide your resume using the available interface.
3. Run the analysis.
4. Review the score, feedback, and suggestions.
5. Update your resume using the relevant recommendations.

## Important Note

The ATS score is an AI-generated estimate, not a result from an employer's actual applicant tracking system. It does not guarantee shortlisting or employment.

Review AI-generated suggestions before applying them. Remove sensitive personal information before submitting a resume, as analysis may send its content to an external AI service.

## Author

**Vairavendhan M**

Aspiring Generative AI professional building practical Python and AI projects.

GitHub: https://github.com/42111392-vendhan
