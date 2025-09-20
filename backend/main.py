import sqlite3
import json
import docx
import pdfplumber
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from thefuzz import fuzz
import nltk
import os
import requests
from dotenv import load_dotenv
import time

# --- NLTK Setup ---
# Checks for NLTK data and provides clear instructions if it's missing.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("NLTK data models found.")
except LookupError:
    print("NLTK data models not found. Attempting to download...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Automatic NLTK download failed: {e}")
        print("Please download the NLTK data manually.")


# --- Configuration ---
load_dotenv()
DATABASE = 'evaluations.db'
GEMINI_API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Skill Lexicon ---
MUST_HAVE_SKILLS = [
    "python", "sql", "machine learning", "deep learning", "data science", "data analysis",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "aws", "azure", "gcp",
    "spark", "pyspark", "databricks", "power bi", "tableau"
]
GOOD_TO_HAVE_SKILLS = [
    "java", "c++", "javascript", "react", "git", "github", "docker", "kubernetes", "api",
    "flask", "django", "fastapi", "statistics", "nlp", "natural language processing",
    "computer vision", "data visualization", "excel", "business intelligence"
]

app = Flask(__name__, static_folder='../frontend', static_url_path='/')

CORS(app)

# --- Database & Text Extraction ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def extract_text(file_stream, filename):
    file_stream.seek(0)
    text = ""
    if filename.endswith('.pdf'):
        try:
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF with pdfplumber: {e}")
    elif filename.endswith('.docx'):
        try:
            doc = docx.Document(file_stream)
            text = "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            print(f"Error reading DOCX: {e}")
    return text.lower()

# --- Analysis Engine with NLTK ---
def perform_weighted_local_analysis(resume_text, jd_text):
    jd_tokens = set(nltk.word_tokenize(jd_text))
    
    jd_must_haves = {skill for skill in MUST_HAVE_SKILLS if skill in jd_tokens or skill in jd_text}
    jd_good_to_haves = {skill for skill in GOOD_TO_HAVE_SKILLS if skill in jd_tokens or skill in jd_text}
    
    if not jd_must_haves and not jd_good_to_haves:
        return 0, [], [], "No relevant technical skills found in the Job Description."

    matched_must_haves = {skill for skill in jd_must_haves if fuzz.partial_ratio(skill, resume_text) > 85}
    matched_good_to_haves = {skill for skill in jd_good_to_haves if fuzz.partial_ratio(skill, resume_text) > 85}
    
    matched_skills = matched_must_haves.union(matched_good_to_haves)
    missing_skills = jd_must_haves.union(jd_good_to_haves) - matched_skills
    
    must_have_score = (len(matched_must_haves) / len(jd_must_haves)) if jd_must_haves else 1.0
    good_to_have_score = (len(matched_good_to_haves) / len(jd_good_to_haves)) if jd_good_to_haves else 1.0
    
    must_have_weight = 0.75
    good_to_have_weight = 0.25
    
    if not jd_must_haves: must_have_weight, good_to_have_weight = 0, 1.0
    if not jd_good_to_haves: must_have_weight, good_to_have_weight = 1.0, 0

    final_score = int((must_have_score * must_have_weight + good_to_have_score * good_to_have_weight) * 100)
    
    feedback = (f"The resume matches {len(matched_skills)} of the {len(jd_must_haves) + len(jd_good_to_haves)} key technical skills required. "
                f"Consider highlighting experience with: {', '.join(sorted(list(missing_skills)))}.")
    
    return final_score, sorted(list(matched_skills)), sorted(list(missing_skills)), feedback


# --- AI Fallback Chain ---
def get_ai_analysis(resume_text, jd_text):
    base_prompt = f"""
    [INST] Analyze the following resume against the job description. Provide your output in a valid JSON format only, with no other text before or after the JSON block.
    The JSON object must have these exact keys: "relevance_score" (an integer from 0 to 100), "matched_skills" (a list of strings), "missing_skills" (a list of strings), and "feedback" (a single string of personalized advice).

    JOB DESCRIPTION: --- {jd_text[:2000]} ---
    RESUME: --- {resume_text[:2000]} --- [/INST]
    """
    if GEMINI_API_KEY:
        try:
            print("Attempting analysis with Google Gemini...")
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": base_prompt}]}]}
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=20)
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text'], "Gemini"
        except Exception as e:
            print(f"Gemini failed: {e}")

    if OPENAI_API_KEY:
        try:
            print("Attempting analysis with OpenAI GPT...")
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = { "model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": base_prompt}], "response_format": {"type": "json_object"} }
            response = requests.post(api_url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'], "OpenAI"
        except Exception as e:
            print(f"OpenAI failed: {e}")

    if HF_TOKEN:
        try:
            print("Attempting analysis with Hugging Face...")
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            payload = {"inputs": base_prompt, "parameters": {"max_new_tokens": 500}}
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 503:
                print("Hugging Face model is loading, please wait...")
                time.sleep(20)
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            full_response_text = response.json()[0]['generated_text']
            json_part = full_response_text.split('[/INST]')[-1].strip()
            return json_part, "HuggingFace"
        except Exception as e:
            print(f"Hugging Face failed: {e}")
    return None, None


# --- Main Flask Routes ---
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files or 'job_description' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    resume_file = request.files['resume']
    jd_file = request.files['job_description']
    resume_text = extract_text(resume_file, resume_file.filename)
    jd_text = extract_text(jd_file, jd_file.filename)

    analysis_result = {}
    analysis_type = ""

    raw_ai_response, ai_provider = get_ai_analysis(resume_text, jd_text)
    
    if raw_ai_response:
        try:
            cleaned_response = raw_ai_response[raw_ai_response.find('{'):raw_ai_response.rfind('}')+1]
            ai_result = json.loads(cleaned_response)
            local_score, _, _, _ = perform_weighted_local_analysis(resume_text, jd_text)
            hybrid_score = int(local_score * 0.4 + ai_result.get('relevance_score', 0) * 0.6)
            
            analysis_result = {
                'relevance_score': hybrid_score,
                'feedback': ai_result.get('feedback', 'No feedback provided.'),
                'matched_skills': json.dumps(ai_result.get('matched_skills', [])),
                'missing_skills': json.dumps(ai_result.get('missing_skills', []))
            }
            analysis_type = f"Hybrid ({ai_provider} + Local)"
            print(f"AI analysis successful with {ai_provider}.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"AI returned invalid data: {e}. Falling back to local analysis.")
            raw_ai_response = None
    
    if not raw_ai_response:
        print("All AI providers failed. Falling back to local analysis.")
        score, matched, missing, feedback = perform_weighted_local_analysis(resume_text, jd_text)
        analysis_result = {
            'relevance_score': score,
            'feedback': feedback,
            'matched_skills': json.dumps(matched),
            'missing_skills': json.dumps(missing)
        }
        analysis_type = "Local Fallback"

    verdict = "Low"
    if analysis_result['relevance_score'] > 75: verdict = "High"
    elif analysis_result['relevance_score'] > 50: verdict = "Medium"

    db = get_db()
    db.execute(
        'INSERT INTO evaluations (resume_filename, jd_filename, relevance_score, verdict, feedback, matched_skills, missing_skills, analysis_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (resume_file.filename, jd_file.filename, analysis_result['relevance_score'], verdict, analysis_result['feedback'], analysis_result['matched_skills'], analysis_result['missing_skills'], analysis_type)
    )
    db.commit()
    return jsonify({**analysis_result, 'verdict': verdict, 'analysis_type': analysis_type})

@app.route('/api/evaluations', methods=['GET'])
def get_evaluations():
    db = get_db()
    search_term = request.args.get('search', '')
    verdict_filter = request.args.get('verdict', '')
    query = 'SELECT * FROM evaluations WHERE (resume_filename LIKE ? OR jd_filename LIKE ?)'
    params = [f'%{search_term}%', f'%{search_term}%']
    if verdict_filter:
        query += ' AND verdict = ?'
        params.append(verdict_filter)
    query += ' ORDER BY id DESC'
    evaluations = db.execute(query, params).fetchall()
    return jsonify([dict(row) for row in evaluations])

@app.route('/api/evaluations/delete', methods=['POST'])
def delete_evaluations():
    db = get_db()
    db.execute('DELETE FROM evaluations')
    db.commit()
    return jsonify({'message': 'History cleared successfully'}), 200

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('backend/schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
        print("Database initialized successfully.")

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)

