# needed imports
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress

# locat the project root
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

# load the data and make sure it loaded correctly
df = pd.read_csv(data_dir / "student_performance_dataset.csv")

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
nColumns = [
    "study_time_hours",
        "attendance_percent",
        "sleep_hours",
        "previous_grade"
]


st.subheader("Descriptive Statistics")

# get the descriptive statistics
numeric_summary = df[numeric_columns].describe().T

# get the median and how skewed the variables are
numeric_summary["median"] = df[numeric_columns].median()
numeric_summary["skewness"] = df[numeric_columns].skew()
st.dataframe(numeric_summary.style.format("{:.4f}"))


# the group-by analysis and aggregation
st.subheader("FInal Exam Scores by Quantitative Variable Groups")

for column in nColumns:
    groups = pd.qcut(df[column], q=4, duplicates="drop")
    summary = df.groupby(groups, observed=True)["final_exam_score"].agg(["count", "mean", "median", "std"])
    st.write(f"Final Exam Scores by {column.replace('_', ' ').title()}")
    st.dataframe(summary.style.format({"count": "{:.0f}", "mean": "{:.4f}", "median": "{:.4f}" , "std": "{:.4f}"}))


# creating the scatterplots
st.subheader("Quantitative Variables vs. Final Exam Score")
for column in nColumns:
    # varibles for the linear regression equation
    slope, intercept, r_value, p_value, std_err = linregress(df[column], df["final_exam_score"])
    x_sort = df[column].sort_values()
    regression_line = slope * x_sort + intercept

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(df[column], df["final_exam_score"], alpha=0.6)

    axis.plot(x_sort, regression_line, color="red", label="Regressino Line")
    equation = f"y = {slope:.4f}x + {intercept:.4f}"
    correlation = f"r = {r_value:.4f}"
    axis.text(0.05, 0.95, f"{equation}\n{correlation}", transform=axis.transAxes, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.8))
    
    axis.set_xlabel(column.replace("_", " ").title())
    axis.set_ylabel("Final Exam Score")
    axis.set_title(f"{column.replace('_', ' ').title()} vs. Final Exam Score")
    axis.legend()
    axis.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# create the r analysis
st.subheader("Correlation Analysis")
correlations = []
for column in nColumns:
    pearson_r, pearson_p = pearsonr(df[column], df["final_exam_score"])
    spearman_r, spearman_p = spearmanr(df[column], df["final_exam_score"])
    correlations.append({
        "Variable": column,
        "Pearson Correlation": pearson_r,
        "Pearson P-Value": pearson_p,
        "Spearman Correlation": spearman_r,
        "Spearman P-Value": spearman_p,
    })
correlation_df = pd.DataFrame(correlations)
st.dataframe(correlation_df.style.format({"Pearson Correlation": "{:.4f}", "Pearson P-Value": "{:.4e}", "Spearman Correlation": "{:.4f}", "Spearman P-Value": "{:.4e}"}))