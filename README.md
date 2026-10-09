# India Data & AI Job Market Analyzer

An interactive analytics project for exploring Data Science, Data Engineering, Machine Learning, and AI job postings in India. It combines job-market analysis, skill extraction from descriptions, salary analysis, and a role-family text classifier.

## Features

- Explore postings by role family and location.
- Extract curated technical skills from job descriptions using phrase matching.
- Compare disclosed annual INR salary ranges by role and skill.
- Predict a job-description role family with TF-IDF and Logistic Regression.
- View the results in a Streamlit dashboard.

## Data source

The project uses the [Indian Job Market Dataset 2025 on Kaggle](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026).

The dataset is a historical snapshot, not a live feed or a complete census of Indian jobs. Download it from Kaggle and place the Excel file here:

```text
data/raw/indian-job-market-dataset-2025.xlsx
```

Raw and processed data are excluded from Git. Check the dataset’s current Kaggle license and terms before sharing or redistributing any data.

## Setup on Windows

Create the virtual environment and install the pinned dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download and place the dataset at the path above. Then run the pipeline:

```powershell
.\.venv\Scripts\python.exe .\src\clean_india_data.py
.\.venv\Scripts\python.exe .\src\extract_skills.py
.\.venv\Scripts\python.exe .\src\train_role_classifier.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard opens locally at `http://localhost:8501`.

## Current results

On the current dataset run:

- 2,285 India/INR Data Science, Data Engineering, ML, and AI postings remained after filtering and deduplication.
- 509 postings had a usable disclosed annual salary range.
- The curated phrase matcher found at least one skill in 1,829 descriptions and produced 7,409 skill mentions.
- The role classifier achieved a macro-F1 of 0.746 on a 351-posting held-out set, compared with 0.139 for a majority-class baseline.

These results describe this dataset and run; they are not live labor-market estimates.

## Method and limitations

- Role-family labels are assigned from job-title rules. The classifier learns to predict these rule-based labels; labels have not been manually reviewed.
- Skill extraction uses a curated phrase dictionary. Coverage is not the same as precision or recall, and mentions may need human review.
- Salary charts use only postings with disclosed annual INR ranges. Salary disclosure may not be representative of all postings.
- The dataset’s posting age is relative text, so the project does not report time-series trends.
- Salary comparisons are descriptive. They do not control for every difference in seniority, location, or employer.

## Project structure

```text
app.py
src/
  clean_india_data.py
  extract_skills.py
  train_role_classifier.py
data/
  raw/          # Downloaded source data; not committed
  processed/    # Generated CSV files; not committed
models/         # Saved classifier
reports/        # Evaluation metrics
requirements.txt
```