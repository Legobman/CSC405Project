# needed imports
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress

# locat the project directories
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
figures_dir = project_dir / "figures"
report_dir = project_dir / "report"

# load the data and make sure it loaded correctly
df_original = pd.read_csv(data_dir / "student_performance_dataset.csv")

# the categorical variables
categorical_columns = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
    "final_grade"
]

# the numeric variables
numeric_columns = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade",
    "final_exam_score"
]

# the numeric variables minus the final exam score
nColumns = [
    "study_time_hours",
        "attendance_percent",
        "sleep_hours",
        "previous_grade"
]

# make a copy of the original database to clean
df = df_original.copy()

# get the basic info we need
dataset_shape = df.shape
dataset_head = df.head()
dataset_info = df.info()
dataset_description = df.describe()

# fetch missing values
missing_before = df.isnull().sum()

# handle the missing data we found from previous tests
df["parental_education"] = df["parental_education"].fillna("MissingInfo")

# check for duplicates
duplicate_rows = df.duplicated().sum()
deplicate_student_ids = df["student_id"].duplicated().sum()

# check for invalid values
invalid_values = {
    "Negative study hours": (df["study_time_hours"] < 0).sum(),
    "Negative sleep hours": (df["sleep_hours"] < 0).sum(),
    "Invalid attendance": ((df["attendance_percent"] < 0) | (df["attendance_percent"] > 100)).sum(),
    "Invalid previous grade": ((df["previous_grade"] < 0) | (df["previous_grade"] > 100)).sum(),
    "Invalid exam scores": ((df["final_exam_score"] < 0) | (df["final_exam_score"] > 100)).sum()
}

# check categorical variable distributions
categorical_distributions = {}
for column in categorical_columns:
    categorical_distributions[column] = df[column].value_counts(dropna=False)

# detect outliers
outliers_results = {}
for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 - 1.5 * iqr
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    outliers_results[column] = {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Count": len(outliers),
        "Values": outliers[column].tolist()
    }

# save the results of the cleaning process
df.to_csv(data_dir / "student_performance_dataset_cleaned.csv", index=False)



# sidebar navigation options
navigation_options =[
    "1. Dataset Overview",
    "2. Exploratory Analysis",
    "3. statistical Analysis",
    "4. Machine Learning Results",
    "5. Visualizations",
    "6. AI-Assisted Analysis",
    "7. Final Conclusion"
]

# sidebar navigation body
section = st.sidebar.radio(
    "choose a report section",
    navigation_options,
    label_visibility="collapsed"
)

# 1. Dataset Overview
if section == "1. Dataset Overview":
    st.title("Dataset Overview")
    st.write("""This project examines the relationship between students' study habits,
                acdemic history, and personal circumstances to their final exam scores.""")
    st.subheader("Research Objective")
    st.write("""The Objective is to investigate which factors in a student's characteristics
                and behaviors are associated with their final exam performance and determine
                which variables are most strongly related to exam scores.""")
    st.subheader("Dataset Information")
    st.write("**Dataset Source**: Kaggle-Student Perfomance and Study Habits Dataset")
    st.write("**Target Variable**: 'final_exam_score'")
    students, attributes = st.columns(2)
    st.write("Number of students", df.shape[0])
    st.write("Number of variables", df.shape[1])
    st.subheader("Variable Information")
    attribute_info = pd.DataFrame({
        "Variable": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Missing Values": df.notna().sum().values
    })

    st.dataframe(attribute_info, hide_index=True)





 