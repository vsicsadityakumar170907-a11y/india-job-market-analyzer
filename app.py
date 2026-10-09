from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
JOBS_PATH = DATA_DIR / "india_target_jobs.csv"
SKILLS_PATH = DATA_DIR / "india_extracted_skills.csv"
MODEL_PATH = ROOT / "models" / "role_classifier.joblib"

st.set_page_config(
    page_title="India Data & AI Job Market",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      h1, h2, h3 { color: #12263a; }
      [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3eaf2;
        padding: 16px;
        border-radius: 12px;
      }
      section[data-testid="stSidebar"] { background: #102a43; }
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] p { color: #f0f4f8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("India Data & AI Job Market")
st.caption(
    "Skills demand, disclosed salary ranges, and role mix · "
    "Historical Naukri listing dataset, not a live job feed"
)

jobs_upload = st.sidebar.file_uploader(
    "Upload processed jobs CSV", type="csv", key="jobs"
)
skills_upload = st.sidebar.file_uploader(
    "Upload extracted skills CSV", type="csv", key="skills"
)

if jobs_upload and skills_upload:
    jobs = pd.read_csv(jobs_upload)
    skills = pd.read_csv(skills_upload)
elif JOBS_PATH.exists() and SKILLS_PATH.exists():
    jobs = pd.read_csv(JOBS_PATH)
    skills = pd.read_csv(SKILLS_PATH)
else:
    st.info("Dashboard ke liye dono processed CSV files upload karein.")
    st.stop()

jobs["job_id"] = jobs["job_id"].astype(str)
skills["job_id"] = skills["job_id"].astype(str)
jobs["salary_midpoint_inr_annual"] = pd.to_numeric(
    jobs["salary_midpoint_inr_annual"], errors="coerce"
)
skills["salary_midpoint_inr_annual"] = pd.to_numeric(
    skills["salary_midpoint_inr_annual"], errors="coerce"
)

st.sidebar.header("Explore postings")

role_options = ["All roles"] + sorted(jobs["role_family"].dropna().unique())
selected_role = st.sidebar.selectbox("Role family", role_options)

top_locations = (
    jobs["location"]
    .fillna("Not listed")
    .value_counts()
    .head(50)
    .index.tolist()
)
location_options = ["All locations"] + top_locations
selected_location = st.sidebar.selectbox("Location", location_options)

filtered_jobs = jobs.copy()
if selected_role != "All roles":
    filtered_jobs = filtered_jobs[
        filtered_jobs["role_family"] == selected_role
    ]
if selected_location != "All locations":
    filtered_jobs = filtered_jobs[
        filtered_jobs["location"].fillna("Not listed") == selected_location
    ]

filtered_skills = skills[
    skills["job_id"].isin(filtered_jobs["job_id"])
].copy()

job_count = filtered_jobs["job_id"].nunique()
salary_count = filtered_jobs["salary_midpoint_inr_annual"].notna().sum()
skill_job_count = filtered_skills["job_id"].nunique()
skill_coverage = skill_job_count / job_count if job_count else 0

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Matching postings", f"{job_count:,}")
metric2.metric("Disclosed annual salaries", f"{salary_count:,}")
metric3.metric("Skill mentions", f"{len(filtered_skills):,}")
metric4.metric("Descriptions with matched skills", f"{skill_coverage:.0%}")

overview_tab, predictor_tab, methodology_tab = st.tabs(
    ["Market overview", "Role predictor", "Methodology"]
)

with overview_tab:
    left, right = st.columns(2)

    with left:
        st.subheader("Most requested skills")
        demand = (
            filtered_skills.groupby("skill")["job_id"]
            .nunique()
            .nlargest(15)
            .sort_values()
            .reset_index(name="postings")
        )
        if demand.empty:
            st.info("Is filter combination ke liye skill matches nahi mile.")
        else:
            figure = px.bar(
                demand,
                x="postings",
                y="skill",
                orientation="h",
                template="plotly_white",
                color_discrete_sequence=["#2563eb"],
                labels={"postings": "Postings", "skill": ""},
            )
            figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figure, use_container_width=True)

    with right:
        st.subheader("Median disclosed salary by role")
        salary_by_role = (
            filtered_jobs.dropna(subset=["salary_midpoint_inr_annual"])
            .groupby("role_family")
            .agg(
                median_salary=("salary_midpoint_inr_annual", "median"),
                salary_postings=("job_id", "nunique"),
            )
            .reset_index()
            .sort_values("median_salary")
        )
        if salary_by_role.empty:
            st.info("Is filter combination ke liye salary data nahi mila.")
        else:
            figure = px.bar(
                salary_by_role,
                x="median_salary",
                y="role_family",
                orientation="h",
                hover_data=["salary_postings"],
                template="plotly_white",
                color_discrete_sequence=["#0f766e"],
                labels={
                    "median_salary": "Median annual salary (INR)",
                    "role_family": "",
                    "salary_postings": "Postings with salary",
                },
            )
            figure.update_xaxes(tickprefix="₹", tickformat=",.0f")
            figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figure, use_container_width=True)

    st.subheader("Salary by commonly listed skill")
    salary_skills = (
        filtered_skills.dropna(subset=["salary_midpoint_inr_annual"])
        .groupby("skill")
        .agg(
            salary_postings=("job_id", "nunique"),
            median_salary=("salary_midpoint_inr_annual", "median"),
        )
        .reset_index()
    )
    salary_skills = salary_skills[salary_skills["salary_postings"] >= 5]
    salary_skills = (
        salary_skills.nlargest(12, "salary_postings")
        .sort_values("median_salary")
    )

    if salary_skills.empty:
        st.info("Filter ke baad salary-skill comparison ke liye data kam hai.")
    else:
        figure = px.bar(
            salary_skills,
            x="median_salary",
            y="skill",
            orientation="h",
            hover_data=["salary_postings"],
            template="plotly_white",
            color_discrete_sequence=["#7c3aed"],
            labels={
                "median_salary": "Median annual salary (INR)",
                "skill": "",
                "salary_postings": "Postings with disclosed salary",
            },
        )
        figure.update_xaxes(tickprefix="₹", tickformat=",.0f")
        figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "Salary charts disclosed annual INR ranges par based hain. "
        "Salary report karne wali postings total postings ka chhota, "
        "possibly biased subset hain; skill salary comparison descriptive hai."
    )

with predictor_tab:
    st.subheader("Predict a role family from a job description")
    st.write(
        "Classifier TF-IDF text features aur Logistic Regression use karta hai. "
        "Prediction ko recruiter ya human review ka replacement na samjhein."
    )

    job_text = st.text_area(
        "Job description",
        height=180,
        placeholder="Yahan job description paste karein...",
    )

    if st.button("Predict role family", type="primary"):
        if len(job_text.strip()) < 40:
            st.warning("Prediction ke liye thoda aur job description paste karein.")
        elif not MODEL_PATH.exists():
            st.error("Model file nahi mili. Pehle training script run karein.")
        else:
            model = joblib.load(MODEL_PATH)
            prediction = model.predict([job_text])[0]
            st.success(f"Predicted role family: **{prediction}**")

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba([job_text]).max()
                st.caption(
                    f"Model score: {confidence:.0%}. "
                    "Yeh score calibrated probability nahi hai."
                )

with methodology_tab:
    st.subheader("How to read this dashboard")
    st.markdown(
        """
        - Source: [Indian Job Market Dataset 2025 on Kaggle](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026).
        - Role groups dataset ke job titles par rules laga kar banaye gaye hain.
        - Skill extraction descriptions mein curated phrase matching se hui hai.
        - Role classifier TF-IDF + Logistic Regression hai; current held-out
          macro-F1 **0.746** tha, majority baseline **0.139**.
        - Model labels title rules se aaye hain aur manually verified nahi hain.
        - Sirf 509 of 2,285 matching postings mein usable annual INR salary mili.
          Isliye salary charts disclosed subset dikhate hain.
        - Dataset ek historical snapshot hai. `jobUploaded` relative text hai,
          isliye dashboard time-series trend claim nahi karta.
        """
    )