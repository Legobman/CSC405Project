# needed imports
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress, ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# the categorical variables minus the final grade letter
cColumns = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job"
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

# fetch missing values
missing_after = df.isnull().sum()

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
    upper = q3 + 1.5 * iqr
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
    "3. Statistical Analysis & Hypothesis Testing",
    "4. Machine Learning Results",
    "5. Visualizations",
    "6. Final Conclusion",
    "7. AI-Assisted Analysis"
]

# sidebar navigation body
section = st.sidebar.radio(
    "choose a report section",
    navigation_options
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

# 2. Exploratory Analysis
elif section == "2. Exploratory Analysis":
    st.title("Exploratory Analysis")

    # data that was missing
    st.header("Missing Data")
    missing_summary = pd.DataFrame({
        "Before cleaning": missing_before,
        "After cleaning": missing_after
    })
    st.dataframe(missing_summary, hide_index=False)
    st.write("""There are missing values in the paraental eduction attribue. 
                Due to the nature of categorical variables, I choose to replace
                these values with 'MissingInfo' instant of throwing them away.""")

    # data that was are outliers
    st.header("Outliers")
    # grab the outliers from the dictionary and display them
    for column, results in outliers_results.items():
        st.write(f"**{column.replace('_', ' ').title()}**")
        st.write(f"Outlier count: {results['Count']}; Outlier values: {results['Values']}")
        st.write(f"")
        st.write(f"Lower Bound: {results['Lower Bound']:.2f}; Upper Bound: {results['Upper Bound']:.2f}")
        st.write(f"")
    st.write("""I have decided that I won't throw away any of these outliers. While a study time of 7.4 or 8.1, it isn't impossible.
                None of the attendance percentage outliers seems so low as to be unreason to be genuine. While close to 3 hours of
                sleep seems extreme, again, it isn't unheard of.""")
    
    # use of descriptive Statistics
    st.header("Descriptive Statistics")

    # quantitative analysis/group-by and aggregation
    st.subheader("Quantitative Analysis")
    # the group-by analysis and aggregation
    st.subheader("FInal Exam Scores by Quantitative Variable Groups")

    for column in nColumns:
        groups = pd.qcut(df[column], q=4, duplicates="drop")
        summary = df.groupby(groups, observed=True)["final_exam_score"].agg(["count", "mean", "median", "std"])
        st.write(f"Final Exam Scores by {column.replace('_', ' ').title()}")
        st.dataframe(summary.style.format({"count": "{:.0f}", "mean": "{:.4f}", "median": "{:.4f}" , "std": "{:.4f}"}))
    
    # categorical analysis/group-by and aggregation
    st.subheader("Categorical Analysis")
    for column in cColumns:
        summary = df.groupby(column, observed=True)["final_exam_score"].agg(["count", "mean", "median", "std"])
        st.write(f"Final Exam Scores by {column.replace('_', ' ').title()}")
        st.dataframe(summary.style.format({"count": "{:.0f}", "mean": "{:.4f}", "median": "{:.4f}" , "std": "{:.4f}"}))
        
    # the descriptive statistics for the numerical attributes
    st.subheader("Numerical Variables")
    descriptive_stats = df[numeric_columns].describe().T
    descriptive_stats["Median"] = df[numeric_columns].median()
    descriptive_stats["Skewness"] = df[numeric_columns].skew()
    st.dataframe(descriptive_stats)

    # the descriptive statistics for the categorical attributes
    st.subheader("Categorical Variables")
    for column in categorical_columns:
        st.write(f"{column.replace('_', ' ').title()}")
        st.dataframe(df[column].value_counts(dropna=False).rename_axis("Category").reset_index(name="Count"),
                     hide_index=True)
# 3. Statistical Analysis & Hypothesis Testing
elif section == "3. Statistical Analysis & Hypothesis Testing":
    st.title("Statistical Analysis & Hypothesis Testing")
    st.write("For all tests, I used an alpha of 0.05")

    # t-tests
    st.header("Independent Samples T-Tests")

    # get the variables that have only two values
    ttest_variables = [
        "gender",
        "internet_access",
        "extracurricular_activities",
        "part_time_job"
    ]
    for column in ttest_variables:
        # fetch the related final exam scores, get the names of the groups, separate them, and perform the t-test
        groups = df.groupby(column)["final_exam_score"].apply(list)
        group = groups.index.tolist()
        group1, group2 = groups.iloc[0], groups.iloc[1]
        statistic, p_value  = ttest_ind(group1, group2, equal_var=False)

        # display the info about the t-test
        st.subheader(f"Exam Scores by {column. replace('_', ' ').title()}")
        st.write(f"**Null Hypothesis**: the mean exam scores are euqal between {group[0]} and {group[1]}")
        st.write(f"**Alternate Hypothesis**: the mean exam scores are different between {group[0]} and {group[1]}")
        st.write("**Reasoning**: This is the most appropriate test when one is comparing the means of two different values.")
        st.write(f"Test Statistic: {statistic:.4f}")
        st.write(f"P-value: {p_value:.4e}")
        if p_value < 0.05:
            st.write("Result: Reject the null hypothesis")
        else:
            st.write("Result: Fail to reject the null hypothesis")
    # One-way ANOVA test
    st.header("One-way ANOVA test")
    
    # the only categorical variable with more then two different values
    column = "parental_education"
    groups = df.groupby(column)["final_exam_score"].apply(list)
    statistic, p_value  = f_oneway(*groups.tolist())
    st.subheader("Exam Scores by Parental Education")
    st.write("**Null Hypothesis**: the mean exam scores are euqal between the different parental education groups")
    st.write("**Alternate Hypothesis**: the mean exam scores are different between the different parental education groups")
    st.write("**Reasoning**: This is the most appropriate test when one is comparing the means of more then two values.")
    st.write(f"Test Statistic: {statistic:.4f}")
    st.write(f"P-value: {p_value:.4e}")
    if p_value < 0.05:
        st.write("Result: Reject the null hypothesis")
    else:
        st.write("Result: Fail to reject the null hypothesis")

    # create the r analysis
    st.subheader("Pearson and Spearman Correlations")
    for column in nColumns:
        pearson_r, pearson_p = pearsonr(df[column], df["final_exam_score"])
        spearman_r, spearman_p = spearmanr(df[column], df["final_exam_score"])

        # display the info about the correlation test
        st.subheader(f"Exam Scores by {column. replace('_', ' ').title()}")
        st.write(f"**Pearson Correlation**")
        st.write(f"**Null Hypothesis**: There is no linear relationship between {column.replace('_', ' ').title()} and final exam scores")
        st.write(f"**Alternate Hypothesis**: There is linear relationship between {column.replace('_', ' ').title()} and final exam scores")
        st.write("**Reasoning**: This is the most appropriate test when one is comparing two different numerical variabels.")
        st.write(f"Test Statistic: {pearson_r:.4f}")
        st.write(f"P-value: {pearson_p:.4e}")
        if pearson_p < 0.05:
            st.write("Result: Reject the null hypothesis. There is a statistically significant linear relationship.")
        else:
            st.write("Result: Fail to reject the null hypothesis. There is not a statistically significant linear relationship.")

        st.write(f"**Spearman Correlation**")
        st.write(f"**Null Hypothesis**: There is no monotonic relationship between {column.replace('_', ' ').title()} and final exam scores")
        st.write(f"**Alternate Hypothesis**: There is monotonic relationship between {column.replace('_', ' ').title()} and final exam scores")
        st.write("**Reasoning**: This is the most appropriate test when one is comparing two different numerical variabels and the relationship might not be linear.")
        st.write(f"Test Statistic: {spearman_r:.4f}")
        st.write(f"P-value: {spearman_p:.4e}")
        if spearman_p < 0.05:
            st.write("Result: Reject the null hypothesis. There is a statistically significant monotonic relationship.")
        else:
            st.write("Result: Fail to reject the null hypothesis. There is not a statistically significant monotonic relationship.")

# 4. Machine Learning Results
elif section == "4. Machine Learning Results":
    st.title("Machine Learning Results")

    # separate predictor variables and the target variable
    x = df[nColumns + cColumns]
    y = df["final_exam_score"]

    # split the data into a 80% training / 20% testing set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    st.write(f"**Training Data**: {X_train.shape[0]}")
    st.write(f"**Testing Data**: {X_test.shape[0]}")
    st.write(f"**Number of Obserations**: {x.shape[0]}")

    # prepare the categorical variables by converting them into numbers so they can be used
    LR_preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), nColumns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cColumns)
    ])
    RF_preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), nColumns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cColumns)
    ])

    # build and train the model pipelines
    LR_model = Pipeline(steps=[
        ("preprocessor", LR_preprocessor),
        ("model", LinearRegression())
    ])
    RF_model = Pipeline(steps=[
        ("preprocessor", RF_preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=1))
    ])
    LR_model.fit(X_train, y_train)
    RF_model.fit(X_train, y_train)

    # generate predictions with test data
    LR_prediction = LR_model.predict(X_test)
    RF_prediction = RF_model.predict(X_test)

    # calculate the mean absolute error, root mean square error, and the r square to judge the models
    LR_MAE = mean_absolute_error(y_test, LR_prediction)
    LR_RMSE = np.sqrt(mean_squared_error(y_test, LR_prediction))
    LR_R2 = r2_score(y_test, LR_prediction)
    RF_MAE = mean_absolute_error(y_test, RF_prediction)
    RF_RMSE = np.sqrt(mean_squared_error(y_test, RF_prediction))
    RF_R2 = r2_score(y_test, RF_prediction)

    # Display the results
    st.subheader("Model Performance Comparison")
    model_results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "MAE": [LR_MAE, RF_MAE],
        "RMSE": [LR_RMSE, RF_RMSE],
        "R^2": [LR_R2, RF_R2]
    })
    st.dataframe(model_results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R^2": "{:.4f}"
        }),
        hide_index=True
    )

    # compare the results
    st.subheader("Model Comparsion")
    if LR_MAE < RF_MAE:
        st.write("**MAE**: Linear Regression has a lower average prediction error")
    else:
        st.write("**MAE**: Random Forest has a lower average prediction error")
    if LR_RMSE < RF_RMSE:
        st.write("**RMSE**: Linear Regression has a lower average prediction error when larger are penalized")
    else:
        st.write("**RMSE**: Random Forest has a lower average prediction error when larger are penalized")
    if LR_R2 > RF_R2:
        st.write("**R^2**: Linear Regression explains more of the variation in the final exam scores")
    else:
        st.write("**R^2**: Random Forest explains more of the variation in the final exam scores")

    # retrieve the feature names and importance scores from the random forest to examine how the model make its predictions
    feature_names = RF_model.named_steps["preprocessor"].get_feature_names_out()
    feature_importance = RF_model.named_steps["model"].feature_importances_
    
    # create a dataframe of these feature
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

    # collapse the different categorical features to make it easier to compare how important the whole variable was
    for column in cColumns:
        compare = importance_df["Feature"].str.startswith(f"cat__{column}_")
        importance_df.loc[compare, "importance"] = importance_df.loc[compare, "Importance"].sum()
        importance_df.loc[compare, "Feature"] = column

    # clear out old duplicate rows from before
    importance_df = importance_df.groupby("Feature", as_index=False)["Importance"].sum()

    # sort the datafram
    importance_df = importance_df.sort_values(by="Importance", ascending=True)

    # display the feature importance
    st.subheader("Random Forest Feature Importance")
    fig, axis = plt.subplots(figsize=(9,6))
    axis.barh(importance_df["Feature"], importance_df["Importance"])
    axis.set_xlabel("Importance Score")
    axis.set_ylabel("Predictor")
    axis.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    st.pyplot(fig)
    fig.savefig(figures_dir / "random_forest_feature_importance.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # summarize the findings from the feature importance
    most_important = importance_df.iloc[-1]
    st.write(f"""**Interpretation**: The most important predictor in the Random Forest model was 
                 {most_important['Feature']}. It has an importance score of {most_important['Importance']:.4f}.
                 The Feature Importance measures the contribution of each predictor in terms of how much it added 
                 to the Random Forest model's predictions with a higher scores meaning that it was more important
                 within the model.""")

# 5. Visualizations
elif section == "5. Visualizations":
    st.title("Visualizations")
    
    st.header("Categorical Variabels and Final Exam Scores")
    
    # make a box plot for the categorical variables
    for column in cColumns:
        fig, axis = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=column, y="final_exam_score", ax=axis)
        axis.set_xlabel(column.replace("_", " ").title())
        axis.set_ylabel("Final Exam Score")
        axis.set_title(f"{column.replace('_', ' ').title()} vs. Final Exam Score")
        axis.grid(True, alpha=0.3)
        plt.xticks(rotation=15)
        st.pyplot(fig)
        fig.savefig(figures_dir / f"{column}_vs_final_exam.jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)
        medians = df.groupby(column, observed=True)["final_exam_score"].median()
        highest = medians.idxmax()
        lowest = medians.idxmin()
        st.write(f"**interpretation**: {highest} has the highest median exam score of {medians.max():.2f}, while {lowest} has the lowest median exam score of {medians.min():.2f}, the boxplots also shows difference in the spread and distribution of scores between groups.")
        
        st.header("Numerical Variabels and Final Exam Scores")
            
    # make a box plot for the categorical variables
    for column in nColumns:
        fig, axis = plt.subplots(figsize=(8, 5))
        sns.regplot(data=df, x=column, y="final_exam_score", ax=axis, scatter_kws={"alpha": 0.6})
        axis.set_xlabel(column.replace("_", " ").title())
        axis.set_ylabel("Final Exam Score")
        axis.set_title(f"{column.replace('_', ' ').title()} vs. Final Exam Score")
        axis.grid(True, alpha=0.3)
        plt.xticks(rotation=15)
        st.pyplot(fig)
        fig.savefig(figures_dir / f"{column}_vs_final_exam.jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)
        correlation, p_value = pearsonr(df[column], df["final_exam_score"])
        if abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        if correlation > 0:
            direction = "positive"
        elif correlation == 0:
            direction = "nuetral"
        else:
            direction = "negative"
        st.write(f"**interpretation**: The scatterplot shows a {strength} {direction} linear relationship betweeen {column.replace('_', ' ')} and final exam scores. (r = {correlation:.4f}, p = {p_value:.4e})")
