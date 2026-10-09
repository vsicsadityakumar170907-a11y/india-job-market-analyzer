from pathlib import Path
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "processed" / "india_target_jobs.csv"
reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

jobs = pd.read_csv(data_path)
jobs = jobs.dropna(subset=["job_description", "role_family"]).copy()
jobs["job_description"] = (
    jobs["job_description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
)
jobs = jobs[jobs["job_description"].ne("")]
jobs = jobs.drop_duplicates(subset=["job_description"])

X = jobs["job_description"]
y = jobs["role_family"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
                stop_words="english",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

baseline_f1 = f1_score(y_test, baseline_predictions, average="macro")
model_f1 = f1_score(y_test, predictions, average="macro")

print(f"Unique descriptions: {len(jobs):,}")
print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
print(f"Majority baseline macro-F1: {baseline_f1:.3f}")
print(f"TF-IDF + Logistic Regression macro-F1: {model_f1:.3f}")
print("\nPer-role results:")
print(classification_report(y_test, predictions, zero_division=0))

report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    reports_dir / "role_classifier_metrics.csv"
)
print("Metrics saved to reports/role_classifier_metrics.csv")
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "role_classifier.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
