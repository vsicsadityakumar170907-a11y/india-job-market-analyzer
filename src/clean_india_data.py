from html import unescape
from pathlib import Path
import re

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "raw" / "indian-job-market-dataset-2025.xlsx"
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(input_path, sheet_name=0)

role_pattern = (
    r"\b(?:data scientist|data science|data analyst|data analytics|"
    r"data engineer|analytics engineer|machine learning|ml engineer|"
    r"artificial intelligence|ai engineer|deep learning)\b"
)

jobs = df[
    df["title"].fillna("").str.contains(role_pattern, case=False, regex=True)
].copy()

before_dedup = len(jobs)
jobs = jobs.drop_duplicates(subset="jobId", keep="first").copy()
duplicates_removed = before_dedup - len(jobs)

# India scope: INR postings only
jobs = jobs[jobs["currency"].eq("INR")].copy()

def role_family(title):
    text = str(title).lower()

    if re.search(r"machine learning|\bml\b|artificial intelligence|\bai\b|deep learning|genai|generative ai", text):
        return "ML / AI"
    if re.search(r"data scientist|data science", text):
        return "Data Scientist"
    if re.search(r"data engineer|analytics engineer|big data", text):
        return "Data Engineer"
    if re.search(r"data analyst|data analytics|business intelligence|\bbi developer\b", text):
        return "Data Analyst / BI"
    return "Other Data Role"

jobs["role_family"] = jobs["title"].map(role_family)

# Remove HTML tags from descriptions and normalize whitespace
jobs["jobDescription"] = (
    jobs["jobDescription"]
    .fillna("")
    .astype(str)
    .map(lambda text: unescape(re.sub(r"<[^>]+>", " ", text)))
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

for column in ["minimumSalary", "maximumSalary"]:
    jobs[column] = pd.to_numeric(jobs[column], errors="coerce")

annual_marker = jobs["salary"].fillna("").astype(str).str.contains(
    r"\bpa\b|per annum|per year|annual|yearly",
    case=False,
    regex=True,
)

salary_valid = (
    jobs["minimumSalary"].gt(0)
    & jobs["maximumSalary"].ge(jobs["minimumSalary"])
    & annual_marker
)

jobs["salary_reported"] = salary_valid
jobs["salary_midpoint_inr_annual"] = (
    jobs["minimumSalary"] + jobs["maximumSalary"]
) / 2

jobs.loc[~salary_valid, "salary_midpoint_inr_annual"] = pd.NA
jobs.loc[~salary_valid, ["minimumSalary", "maximumSalary"]] = pd.NA

for column in ["minimumExperience", "maximumExperience"]:
    jobs.loc[jobs[column].eq(0), column] = pd.NA

jobs = jobs.rename(
    columns={
        "jobId": "job_id",
        "companyName": "company_name",
        "jobUploaded": "job_uploaded",
        "tagsAndSkills": "tags_and_skills",
        "minimumSalary": "minimum_salary_inr_annual",
        "maximumSalary": "maximum_salary_inr_annual",
        "minimumExperience": "minimum_experience_years",
        "maximumExperience": "maximum_experience_years",
        "jobDescription": "job_description",
        "ReviewsCount": "reviews_count",
        "AggregateRating": "company_rating",
    }
)

columns = [
    "job_id",
    "title",
    "role_family",
    "company_name",
    "location",
    "job_uploaded",
    "experience",
    "minimum_experience_years",
    "maximum_experience_years",
    "salary",
    "minimum_salary_inr_annual",
    "maximum_salary_inr_annual",
    "salary_midpoint_inr_annual",
    "salary_reported",
    "tags_and_skills",
    "job_description",
]

output_path = output_dir / "india_target_jobs.csv"
jobs[columns].to_csv(output_path, index=False)

print("Target INR postings saved:", len(jobs))
print("Duplicate target rows removed:", duplicates_removed)
print("Rows with disclosed annual salary:", int(jobs["salary_reported"].sum()))
print("\nPostings by role family:")
print(jobs["role_family"].value_counts().to_string())
print("\nSaved:", output_path)