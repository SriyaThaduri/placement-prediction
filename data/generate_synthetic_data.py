import json
import random
import csv

def load_skills_data():
    with open('skills.json', 'r') as file:
        skills_data = json.load(file)
    return skills_data

def calculate_technical_skills_score(user_skills, user_domain, projects, skills_data):
    user_skills_list = [skill.strip().lower() for skill in user_skills.split(',')]
    crucial_skills = skills_data['domain_critical_skills'].get(user_domain, [])
    crucial_skills_score = 0
    project_skills = set()
    for project in projects:
        for skill in project['skills']:
            project_skills.add(skill.strip().lower())
    for skill in user_skills_list:
        skill_score = skills_data['technical_skills'].get(skill, 0)
        if skill in crucial_skills:
            if skill in project_skills:
                crucial_skills_score += skill_score * 1.5
            else:
                crucial_skills_score += skill_score
    selected_domain_skills = skills_data['domain_critical_skills'][user_domain]
    max_possible_score = sum(
        skills_data['technical_skills'][skill] for skill in selected_domain_skills
    ) * 1.5
    final_technical_score_percentage = (crucial_skills_score / max_possible_score) * 100
    return final_technical_score_percentage

def calculate_soft_skills_score(user_soft_skills, skills_data):
    user_soft_skills_list = [skill.strip().lower() for skill in user_soft_skills.split(',')]
    soft_skills_score = 0
    for skill in user_soft_skills_list:
        skill_score = skills_data['soft_skills'].get(skill, 0)
        soft_skills_score += skill_score
    return soft_skills_score

def generate_project_skills(skills_data):
    domains = list(skills_data['domain_critical_skills'].keys())
    user_domain = random.choice(domains)
    crucial_skills = skills_data['domain_critical_skills'][user_domain]
    num_projects = random.randint(1, 3)
    projects = []
    for _ in range(num_projects):
        num_skills = random.randint(2, 5)
        skills = random.sample(crucial_skills + list(skills_data['technical_skills'].keys()), num_skills)
        projects.append({'title': f'Project {_+1}', 'skills': skills})
    return user_domain, projects

def generate_synthetic_data(num_samples=1000):
    skills_data = load_skills_data()
    domains = list(skills_data['domain_critical_skills'].keys())
    soft_skills = list(skills_data['soft_skills'].keys())

    with open('data/training_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['cgpa', 'final_technical_score_percentage', 'soft_skills_score',
                      'aptitude_test_score', 'internship_experience', 'user_domain',
                      'placement', 'estimated_package']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(num_samples):
            cgpa = round(random.uniform(6.0, 10.0), 2)
            user_soft_skills = random.sample(soft_skills, random.randint(2, 5))
            soft_skills_score = calculate_soft_skills_score(','.join(user_soft_skills), skills_data)
            aptitude_test_score = random.randint(50, 100)
            internship_experience = random.randint(0, 1)
            user_domain, projects = generate_project_skills(skills_data)
            user_skills = random.sample(list(skills_data['technical_skills'].keys()), random.randint(3, 7))
            final_technical_score_percentage = calculate_technical_skills_score(
                ','.join(user_skills), user_domain, projects, skills_data
            )
            # Simulate placement based on some heuristic
            placement_prob = final_technical_score_percentage * 0.6 + aptitude_test_score * 0.3 + soft_skills_score * 0.1
            placement = 1 if placement_prob > 75 else 0
            estimated_package = round(random.uniform(30000, 100000), 2) if placement == 1 else 0
            writer.writerow({
                'cgpa': cgpa,
                'final_technical_score_percentage': final_technical_score_percentage,
                'soft_skills_score': soft_skills_score,
                'aptitude_test_score': aptitude_test_score,
                'internship_experience': internship_experience,
                'user_domain': user_domain,
                'placement': placement,
                'estimated_package': estimated_package
            })

    print("Synthetic training data generated at data/training_data.csv")
if __name__ == "__main__":
    generate_synthetic_data(num_samples=1000)
    print("Synthetic training data generated at data/training_data.csv")