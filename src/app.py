# needed imports
import pandas as pd
from pathlib import Path

# locat the project root
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

# load the data and make sure it loaded correctly
df = pd.read_csv(data_dir / "student_performance_dataset.csv")
print("Dataset shape: ", df.shape)
print(df.head())
df.info()
print(df.describe())

# check for invalid data
print("Missing values:")
print(df.isnull().sum())
df["parental_education"] = df["parental_education"].fillna("MissingInfo")
print(df.isnull().sum())
print("Duplicate rows: ", df.duplicated().sum())
print("Duplicate student IDs: ", df["student_id"].duplicated().sum())
print("Negative study hours: ", (df["study_time_hours"] < 0).sum())
print("Negative sleep hours: ", (df["sleep_hours"] < 0).sum())
print("Invalid attendance: ", ((df["attendance_percent"] < 0) | (df["attendance_percent"] > 100)).sum())
print("Invalid previous grade: ", ((df["previous_grade"] < 0) | (df["previous_grade"] > 100)).sum())
print("Invalid exam scores: ", ((df["final_exam_score"] < 0) | (df["final_exam_score"] > 100)).sum())

# display the categorical variables and how often each value appears
categorical_columns = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
    "final_grade"
]
for column in categorical_columns:
    print(df[column].value_counts(dropna=False))
    print()

# display the numeric variables
numeric_columns = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade",
    "final_exam_score"
]

# calculate the quantiles and find any outliers
for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(column, "- outliers: ", len(outliers))
    print(outliers[column].values)

# record the cleaned up data to a new file
df.to_csv(data_dir / "student_performance_dataset_cleaned.csv")