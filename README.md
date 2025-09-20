Automated Resume Relevance Check System

**Live Application URL:** [https://resume-analyzer-a91y.onrender.com](https://www.google.com/search?q=https://resume-analyzer-a91y.onrender.com)

An intelligent web application designed to automate the process of screening resumes against job descriptions. This tool calculates a relevance score, provides a suitability verdict, and offers actionable feedback, helping recruitment teams to quickly and consistently identify the most qualified candidates.

## **Problem Statement**

Recruitment teams often face a high volume of applications for each job opening. The manual process of reviewing each resume against a job description is:

* **Time-Consuming:** It leads to significant delays in the shortlisting process.  
* **Inconsistent:** Different evaluators may interpret requirements differently, leading to biased judgments.  
* **High Workload:** It reduces the time staff can spend on more value-added tasks like interview preparation and student guidance.

This project solves this by providing a scalable, consistent, and automated system for the initial screening phase.

## **Our Approach & Solution**

This project is a full-stack web application with a Python/Flask backend and an HTML/Tailwind CSS frontend. The core of the application is a sophisticated **local AI analysis engine**.

Our initial goal was to build a hybrid system combining local analysis with a live "Semantic Match" from a Generative AI model (like Gemini or GPT). However, we encountered persistent network-level restrictions that blocked calls to external AI services.

In response, we engineered a robust and powerful local AI engine that works entirely offline, demonstrating resilience and a commitment to delivering a functional product despite external constraints.

### **Key Features:**

* **Intelligent Local AI:** Uses the **Natural Language Toolkit (NLTK)** to parse documents and understand their content.  
* **Weighted Skill Scoring:** Differentiates between "must-have" and "good-to-have" skills to produce a more accurate and nuanced relevance score.  
* **Fuzzy Matching:** Catches minor typos and variations in skill names (e.g., "scikit learn" vs. "scikit-learn").  
* **PDF & DOCX Support:** Accepts resumes and job descriptions in the two most common formats.  
* **Persistent History:** Automatically saves every analysis to a database.  
* **Interactive Dashboard:** Allows recruiters to view, search, and filter the entire evaluation history.  
* **AI-Ready:** The backend is engineered with a fallback chain. If a network environment allows it, the system can be instantly upgraded by adding API keys to the .env file with zero code changes.

## **Installation and Local Setup**

Follow these steps to run the project on your local machine.

### **Prerequisites**

* Python 3.11 (This is recommended to ensure compatibility with all dependencies).  
* A Git client.

### **1\. Clone the Repository**

git clone \[https://github.com/krsnassuhrt/Resume-Analyzer.git\](https://github.com/krsnassuhrt/Resume-Analyzer.git)  
cd Resume-Analyzer

### **2\. Set Up the Python Virtual Environment**

It is highly recommended to use a virtual environment. We recommend using Python 3.11.

\# Create a virtual environment using Python 3.11  
py \-3.11 \-m venv .venv

\# Activate the environment  
\# On Windows:  
.venv\\Scripts\\activate  
\# On macOS/Linux:  
\# source .venv/bin/activate

### **3\. Install Dependencies**

All required libraries are listed in the requirements.txt file.

\# Navigate to the backend folder  
cd backend

\# Install the dependencies  
pip install \-r requirements.txt

### **4\. Set Up Environment Variables (Optional)**

The application is designed to attempt connections to live AI models. If you have API keys, you can add them.

1. In the backend folder, create a new file named .env.  
2. Add your keys to this file. If you don't have a key, you can leave the value blank.  
   API\_KEY=your\_google\_gemini\_api\_key  
   OPENAI\_API\_KEY=your\_openai\_api\_key  
   HF\_TOKEN=your\_hugging\_face\_api\_key

   **Note:** The application will fall back to the local NLTK engine if the keys are missing or if the connections fail.

### **5\. Run the Application**

1. Make sure you are in the backend directory with your virtual environment active.  
2. Run the Flask server:  
   python main.py  
