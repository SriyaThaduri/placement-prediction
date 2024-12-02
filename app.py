import json
import os
from flask import Flask, render_template, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import joblib
import numpy as np
import pandas as pd
#from dotenv import load_dotenv
from functools import wraps

# Load environment variables from .env file
#load_dotenv()

app = Flask(__name__)
DATABASE_URL=" "

# Configuration for Supabase PostgreSQL database
#app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Load the trained classification models and scaler
clf_model = joblib.load('models/placement_model.pkl')          # Logistic Regression
rf_clf_model = joblib.load('models/rf_placement_model.pkl')  # Random Forest Classifier
preprocessor = joblib.load('models/preprocessor.joblib')      # Preprocessor

# Load the trained regression model
reg_model = joblib.load('models/salary_model.pkl')

# Load skills data
def load_skills_data():
    with open('skills.json', 'r') as file:
        skills_data = json.load(file)
    return skills_data

skills_data = load_skills_data()

# Database Model
class UserResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(50), nullable=False)
    cgpa = db.Column(db.Float, nullable=False)
    user_skills = db.Column(db.String(500), nullable=False)
    user_soft_skills = db.Column(db.String(500), nullable=False)
    aptitude_test_score = db.Column(db.Float, nullable=False)
    internship_experience = db.Column(db.Boolean, nullable=False)
    user_domain = db.Column(db.String(50), nullable=False)
    recommended_domain = db.Column(db.String(50), nullable=False)
    placement_prediction = db.Column(db.Boolean, nullable=False)
    estimated_package = db.Column(db.Float, nullable=False)
    missing_skills = db.Column(db.String(500), nullable=True)
    low_scoring_skills = db.Column(db.String(500), nullable=True)
    missing_soft_skills = db.Column(db.String(500), nullable=True)
    projects = db.Column(db.Text, nullable=True)  # Store projects as JSON string

    def __repr__(self):
        return f'<UserResult {self.roll_number}>'

# Authentication Credentials (For demonstration; use secure methods in production)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password123'

def check_auth(username, password):
    """Check if a username/password combination is valid."""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def authenticate():
    """Sends a 401 response that enables basic auth."""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    """Decorator to prompt for user authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Function Definitions
def calculate_technical_skills_score(user_skills, user_domain, projects):
    # Parse user skills
    user_skills_list = [skill.strip().lower() for skill in user_skills.split(',')]

    # Get the crucial skills for the user's domain
    crucial_skills = skills_data['domain_critical_skills'].get(user_domain, [])

    crucial_skills_score = 0

    # Extract project skills
    project_skills = set()
    for project in projects:
        for skill in project['skills']:
            project_skills.add(skill.strip().lower())

    for skill in user_skills_list:
        # Add to total technical score
        skill_score = skills_data['technical_skills'].get(skill, 0)

        # Check if the skill is crucial for the selected domain
        if skill in crucial_skills:
            # Only add increased weight if the skill is also mentioned in projects
            if skill in project_skills:
                # Increased weight for domain-critical skills
                crucial_skills_score += skill_score * 1.5
            else:
                crucial_skills_score += skill_score

    # Final technical score, with extra weight given to crucial skills mentioned in projects
    final_technical_score = crucial_skills_score

    # Convert the final technical score to a percentage
    selected_domain_skills = skills_data['domain_critical_skills'][user_domain]
    max_possible_score = sum(
        skills_data['technical_skills'][skill] for skill in selected_domain_skills) * 1.5

    final_technical_score_percentage = (
        final_technical_score / max_possible_score) * 100

    return final_technical_score_percentage

def calculate_soft_skills_score(user_soft_skills):
    # Parse user soft skills
    user_soft_skills_list = [skill.strip().lower() for skill in user_soft_skills.split(',')]

    soft_skills_score = 0

    for skill in user_soft_skills_list:
        # Add soft skills scores based on the predefined weights
        skill_score = skills_data['soft_skills'].get(skill, 0)
        soft_skills_score += skill_score

    return soft_skills_score

def recommend_missing_skills(user_skills, user_domain, projects):
    user_skills_set = set([skill.strip().lower() for skill in user_skills.split(',')])
    crucial_skills = set([skill.lower() for skill in skills_data['domain_critical_skills'].get(user_domain, [])])

    missing_skills = list(crucial_skills - user_skills_set)

    project_skills_set = set()
    for project in projects:
        for skill in project['skills']:
            project_skills_set.add(skill.strip().lower())

    skills_not_in_projects = list(user_skills_set - project_skills_set)

    return missing_skills, skills_not_in_projects

def recommend_missing_soft_skills(user_soft_skills):
    user_soft_skills_set = set([skill.strip().lower() for skill in user_soft_skills.split(',')])
    all_soft_skills = set([skill.lower() for skill in skills_data['soft_skills'].keys()])

    missing_soft_skills = list(all_soft_skills - user_soft_skills_set)

    return missing_soft_skills

def recommend_domain(user_skills, user_domain, projects):
    user_skills_set = set([skill.strip().lower() for skill in user_skills.split(',')])

    # Calculate scores for each domain based on critical skills
    domain_scores = {}
    for domain, critical_skills in skills_data['domain_critical_skills'].items():
        critical_skills_set = set([skill.lower() for skill in critical_skills])
        matched_skills = user_skills_set.intersection(critical_skills_set)
        domain_scores[domain] = len(matched_skills)

    # Recommend the domain with the highest score
    recommended_domain = max(domain_scores, key=domain_scores.get)

    return recommended_domain

# Prediction Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve form data
            user_name = request.form['user_name']
            roll_number = request.form['roll_number']
            cgpa = float(request.form['cgpa'])
            user_skills = request.form['user_skills']
            user_soft_skills = request.form['soft_skills']
            aptitude_test_score = float(request.form['aptitude_test_score'])
            internship_experience = True if request.form.get('internship_experience') == 'yes' else False
            user_domain = request.form['user_domain'].lower()

            # Retrieve projects data
            projects = []
            project_titles = request.form.getlist('project_title')
            project_skills = request.form.getlist('project_skills')

            for title, skills in zip(project_titles, project_skills):
                projects.append({
                    'title': title,
                    'skills': [skill.strip().lower() for skill in skills.split(',')]
                })

            # Calculate technical skills score with projects
            final_technical_score = calculate_technical_skills_score(
                user_skills, user_domain, projects
            )

            # Calculate soft skills score
            soft_skills_score = calculate_soft_skills_score(user_soft_skills)

            # Recommend a suitable domain
            recommended_domain = recommend_domain(user_skills, user_domain, projects)

            # Prepare input features for the model
            input_features = {
                'cgpa': cgpa,
                'final_technical_score_percentage': final_technical_score,
                'soft_skills_score': soft_skills_score,
                'aptitude_test_score': aptitude_test_score,
                'internship_experience': int(internship_experience),
                'user_domain': recommended_domain
            }

            input_df = pd.DataFrame([input_features])

            # Preprocess the input features
            input_features_scaled = preprocessor.transform(input_df)

            # Predict placement using Logistic Regression
            placement_prediction = clf_model.predict(input_features_scaled)[0]

            # Predict estimated salary
            if placement_prediction == 1:
                estimated_package = reg_model.predict(input_features_scaled)[0]
                estimated_package = round(estimated_package, 2)
            else:
                estimated_package = 0

            # Recommend missing and low-scoring skills
            missing_skills, skills_not_in_projects = recommend_missing_skills(
                user_skills, recommended_domain, projects
            )
            missing_soft_skills = recommend_missing_soft_skills(user_soft_skills)

            # Store the result in the database
            user_result = UserResult(
                user_name=user_name,
                roll_number=roll_number,
                cgpa=cgpa,
                user_skills=user_skills,
                user_soft_skills=user_soft_skills,
                aptitude_test_score=aptitude_test_score,
                internship_experience=internship_experience,
                user_domain=user_domain,
                recommended_domain=recommended_domain,
                placement_prediction=bool(placement_prediction),
                estimated_package=estimated_package,
                missing_skills=json.dumps(missing_skills),
                low_scoring_skills=json.dumps(skills_not_in_projects),
                missing_soft_skills=json.dumps(missing_soft_skills),
                projects=json.dumps(projects)
            )

            db.session.add(user_result)
            db.session.commit()

            return render_template('result.html',
                                   user_name=user_name,
                                   roll_number=roll_number,
                                   placement=placement_prediction,
                                   estimated_package=estimated_package,
                                   missing_skills=missing_skills,
                                   low_scoring_skills=skills_not_in_projects,
                                   missing_soft_skills=missing_soft_skills,
                                   recommended_domain=recommended_domain,
                                   projects=projects)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', error="An error occurred during processing. Please check your inputs.")
    return render_template('index.html')

# Admin Route with Data Processing
@app.route('/admin')
def admin():
    try:
        # Fetch all user results from the database
        all_results = UserResult.query.all()
        
        # Prepare data for the template
        results = []
        for result in all_results:
            # Deserialize JSON-encoded fields
            missing_skills = json.loads(result.missing_skills) if result.missing_skills else []
            low_scoring_skills = json.loads(result.low_scoring_skills) if result.low_scoring_skills else []
            missing_soft_skills = json.loads(result.missing_soft_skills) if result.missing_soft_skills else []
            projects = json.loads(result.projects) if result.projects else []
            
            # Append processed data to the results list
            results.append({
                'user_name': result.user_name,
                'roll_number': result.roll_number,
                'cgpa': result.cgpa,
                'user_skills': result.user_skills.split(',') if result.user_skills else [],
                'user_soft_skills': result.user_soft_skills.split(',') if result.user_soft_skills else [],
                'aptitude_test_score': result.aptitude_test_score,
                'internship_experience': result.internship_experience,
                'user_domain': result.user_domain,
                'recommended_domain': result.recommended_domain,
                'placement_prediction': result.placement_prediction,
                'estimated_package': result.estimated_package,
                'missing_skills': missing_skills,
                'low_scoring_skills': low_scoring_skills,
                'missing_soft_skills': missing_soft_skills,
                'projects': projects
            })
        
        return render_template('admin.html', results=results)
    
    except Exception as e:
        # Log the error (Enhance logging as needed)
        print(f"Error fetching admin data: {e}")
        return render_template('admin.html', results=[], error="An error occurred while fetching data.")

if __name__ == "__main__":
    app.run(debug=True, port=3000)