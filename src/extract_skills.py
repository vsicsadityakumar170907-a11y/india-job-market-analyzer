from pathlib import Path
import re

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "processed" / "india_target_jobs.csv"
output_dir = project_root / "data" / "processed"

jobs = pd.read_csv(input_path)

skill_aliases = {
    "Python": ["python"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sql server", "sqlite"],
    "Java": ["java"],
    "JavaScript": ["javascript"],
    "C++": ["c++"],
    "Excel": ["excel", "microsoft excel"],
    "Statistics": ["statistics", "statistical"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
    "Generative AI": ["generative ai", "genai"],
    "LLM": ["large language model", "llm", "llms"],
    "NLP": ["natural language processing", "nlp"],
    "RAG": ["retrieval augmented generation", "rag"],
    "Computer Vision": ["computer vision"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Apache Spark": ["apache spark", "pyspark", "spark"],
    "Hadoop": ["hadoop"],
    "Airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt"],
    "Databricks": ["databricks"],
    "AWS": ["amazon web services", "aws"],
    "Azure": ["microsoft azure", "azure"],
    "Google Cloud": ["google cloud platform", "google cloud", "gcp"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "MLflow": ["mlflow"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "Git": ["git", "github"],
}

patterns = {}
for skill, aliases in skill_aliases.items():
    aliases = sorted(aliases, key=len, reverse=True)
    alternatives = "|".join(re.escape(alias) for alias in aliases)
    patterns[skill] = re.compile(
        rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )

skill_rows = []

for row in jobs.itertuples(index=False):
    description = str(row.job_description or "")
    for skill, pattern in patterns.items():
        if pattern.search(description):
            skill_rows.append(
                {
                    "job_id": row.job_id,
                    "title": row.title,
                    "role_family": row.role_family,
                    "location": row.location,
                    "salary_midpoint_inr_annual": row.salary_midpoint_inr_annual,
                    "skill": skill,
                }
            )

extracted = pd.DataFrame(skill_rows)
output_path = output_dir / "india_extracted_skills.csv"
extracted.to_csv(output_path, index=False)

jobs_with_skills = extracted["job_id"].nunique() if not extracted.empty else 0

print("Target job postings:", len(jobs))
print("Postings with at least one extracted skill:", jobs_with_skills)
print("Unique skill mentions:", len(extracted))
print("\nTop skills found in descriptions:")
if not extracted.empty:
    print(extracted["skill"].value_counts().head(20).to_string())

print("\nSaved:", output_path)