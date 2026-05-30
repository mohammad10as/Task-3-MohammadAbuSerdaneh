import math
import json
from collections import Counter

JOB_ROLES = [
    {
        "title": "Data Scientist",
        "skills": ["python", "machine learning", "sql", "statistics", "data analysis",
                   "tensorflow", "pandas", "numpy", "deep learning", "visualization"]
    },
    {
        "title": "DevOps Engineer",
        "skills": ["aws", "docker", "kubernetes", "ci/cd", "linux", "git",
                   "automation", "cloud", "terraform", "monitoring"]
    },
    {
        "title": "Backend Developer",
        "skills": ["python", "java", "sql", "apis", "rest", "databases",
                   "node", "django", "flask", "microservices"]
    },
    {
        "title": "Frontend Developer",
        "skills": ["javascript", "html", "css", "react", "vue",
                   "typescript", "ui", "ux", "responsive design", "web"]
    },
    {
        "title": "Machine Learning Engineer",
        "skills": ["python", "machine learning", "deep learning", "tensorflow",
                   "pytorch", "mlops", "docker", "cloud", "algorithms", "optimization"]
    },
    {
        "title": "Cloud Architect",
        "skills": ["aws", "azure", "cloud", "automation", "terraform",
                   "kubernetes", "security", "networking", "docker", "devops"]
    },
    {
        "title": "Cybersecurity Analyst",
        "skills": ["security", "networking", "linux", "penetration testing",
                   "firewalls", "encryption", "monitoring", "incident response", "python", "risk analysis"]
    },
    {
        "title": "Full Stack Developer",
        "skills": ["javascript", "python", "react", "node", "sql",
                   "html", "css", "apis", "databases", "git"]
    },
    {
        "title": "Data Engineer",
        "skills": ["python", "sql", "spark", "hadoop", "etl",
                   "data pipelines", "cloud", "aws", "databases", "kafka"]
    },
    {
        "title": "AI Research Scientist",
        "skills": ["python", "deep learning", "machine learning", "mathematics",
                   "statistics", "pytorch", "tensorflow", "algorithms", "research", "optimization"]
    },
    {
        "title": "Mobile Developer",
        "skills": ["java", "kotlin", "swift", "react native", "flutter",
                   "mobile", "ui", "apis", "databases", "git"]
    },
    {
        "title": "Systems Administrator",
        "skills": ["linux", "windows", "networking", "automation", "scripting",
                   "monitoring", "security", "cloud", "virtualization", "troubleshooting"]
    },
]


def compute_tf(skill_list):
    """Term Frequency: count of term / total terms in document"""
    total = len(skill_list)
    counts = Counter(skill_list)
    return {term: count / total for term, count in counts.items()}


def compute_idf(all_documents):
    """Inverse Document Frequency: log(total docs / docs containing term)"""
    total_docs = len(all_documents)
    term_doc_count = {}

    for doc in all_documents:
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_count[term] = term_doc_count.get(term, 0) + 1

    idf = {}
    for term, count in term_doc_count.items():
        idf[term] = math.log(total_docs / count)

    return idf


def compute_tfidf_vector(skill_list, idf, vocabulary):
    """Create a TF-IDF weighted vector for a skill list"""
    tf = compute_tf(skill_list)
    vector = []
    for term in vocabulary:
        tfidf_score = tf.get(term, 0) * idf.get(term, 0)
        vector.append(tfidf_score)
    return vector


def cosine_similarity(vec_a, vec_b):
    """
    cos(θ) = (A · B) / (||A|| × ||B||)
    Returns a value between 0 and 1
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold Start: zero vector → no similarity

    return dot_product / (magnitude_a * magnitude_b)


def build_recommender():
    """Pre-compute IDF and vocabulary from the dataset"""
    all_skill_lists = [role["skills"] for role in JOB_ROLES]

    # Build shared vocabulary
    vocabulary = sorted(set(skill for skills in all_skill_lists for skill in skills))

    # Compute IDF across all job role documents
    idf = compute_idf(all_skill_lists)

    # Pre-compute TF-IDF vectors for all job roles
    role_vectors = []
    for role in JOB_ROLES:
        vec = compute_tfidf_vector(role["skills"], idf, vocabulary)
        role_vectors.append(vec)

    return vocabulary, idf, role_vectors


def recommend(user_skills_raw, top_n=3):
    """
    Full 4-Step Pipeline:
    1. Ingestion  → parse and normalize user skills
    2. Scoring    → cosine similarity against all job roles
    3. Sorting    → descending by score
    4. Filtering  → return Top-N results
    """
    # Step 1: INGESTION — normalize input
    user_skills = [skill.strip().lower() for skill in user_skills_raw]

    if len(user_skills) < 3:
        return {"error": "Please provide at least 3 skills for accurate matching."}

    vocabulary, idf, role_vectors = build_recommender()

    user_vector = compute_tfidf_vector(user_skills, idf, vocabulary)

    if all(v == 0 for v in user_vector):
        return {
            "error": "None of your skills matched our dataset vocabulary. Try skills like: Python, AWS, Docker, SQL, React, Machine Learning..."
        }

    scored_roles = []
    for i, role in enumerate(JOB_ROLES):
        score = cosine_similarity(user_vector, role_vectors[i])
        scored_roles.append({
            "title": role["title"],
            "score": round(score, 4),
            "match_percent": round(score * 100, 1),
            "role_skills": role["skills"]
        })

    scored_roles.sort(key=lambda x: x["score"], reverse=True)

    top_results = scored_roles[:top_n]

    return {
        "user_skills": user_skills,
        "recommendations": top_results,
        "total_roles_evaluated": len(JOB_ROLES)
    }


try:
    from flask import Flask, request, jsonify, send_file
    import os

    app = Flask(__name__)

    @app.route("/")
    def index():
        return send_file("index.html")

    @app.route("/recommend", methods=["POST"])
    def get_recommendations():
        data = request.get_json()
        skills = data.get("skills", [])
        top_n = data.get("top_n", 3)
        result = recommend(skills, top_n)
        return jsonify(result)

    @app.route("/skills-list", methods=["GET"])
    def skills_list():
        """Return all available skills for autocomplete"""
        all_skills = sorted(set(
            skill for role in JOB_ROLES for skill in role["skills"]
        ))
        return jsonify(all_skills)

    if __name__ == "__main__":
        print("\n🚀 Tech Stack Recommender is running!")
        print("📡 Open http://localhost:5000 in your browser\n")
        app.run(debug=True, port=5000)

except ImportError:
    if __name__ == "__main__":
        print("=== Tech Stack Recommender (CLI Mode) ===\n")
        print("Enter your skills separated by commas:")
        raw = input("> ")
        skills = [s.strip() for s in raw.split(",")]
        result = recommend(skills)

        if "error" in result:
            print(f"\n❌ {result['error']}")
        else:
            print(f"\n✅ Top Recommendations for: {', '.join(result['user_skills'])}")
            print(f"   (Evaluated {result['total_roles_evaluated']} job roles)\n")
            for i, rec in enumerate(result["recommendations"], 1):
                bar = "█" * int(rec["match_percent"] / 5)
                print(f"  {i}. {rec['title']}")
                print(f"     Match: {rec['match_percent']}% {bar}")
                print()
