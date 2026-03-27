import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    r2_score,
    mean_squared_error,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASS_CATEGORICAL = [
    "Q1_Age_Group", "Q2_Gender", "Q3_Employment", "Q4_Income", "Q5_Pet_Type",
    "Q5b_Ownership_Duration", "Q6_Boarding_Frequency", "Q7_Boarding_Reason",
    "Q11_Urgency", "Q15_Care_Preference", "Q17_Competitive_Awareness",
    "Q18_Switching_Threshold", "Q20_Payment_Model", "Q24_Adoption_Barrier"
]
CLASS_NUMERIC = [
    "Q9_Monthly_Pet_Spend_USD", "Derived_Attachment_Index", "Derived_Concern_Index",
    "Derived_Trust_Index", "Derived_TPB_Composite", "Derived_Satisfaction_Index"
]

REG_CATEGORICAL = [
    "Q1_Age_Group", "Q3_Employment", "Q4_Income", "Q5_Pet_Type",
    "Q6_Boarding_Frequency", "Q11_Urgency", "Q15_Care_Preference",
    "Q20_Payment_Model", "Q24_Adoption_Barrier"
]
REG_NUMERIC = [
    "Q9_Monthly_Pet_Spend_USD", "Derived_Attachment_Index", "Derived_Concern_Index",
    "Derived_Trust_Index", "Derived_TPB_Composite", "Derived_Satisfaction_Index"
]

CLUSTER_FEATURES = [
    "Derived_Attachment_Index", "Derived_Concern_Index", "Derived_Trust_Index",
    "Derived_TPB_Composite", "Derived_Satisfaction_Index", "Q9_Monthly_Pet_Spend_USD",
    "Derived_PSM_WTP_Midpoint"
]

def _build_preprocessor(cat_cols, num_cols):
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])

def _get_feature_names(preprocessor, cat_cols, num_cols):
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_cols)
    )
    return cat_names + list(num_cols)

def run_classification(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["Adopt_Binary"] = (work["Q25_Adoption_Intent"] == "Yes").astype(int)
    X = work[CLASS_CATEGORICAL + CLASS_NUMERIC]
    y = work["Adopt_Binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log_model = Pipeline([
        ("pre", _build_preprocessor(CLASS_CATEGORICAL, CLASS_NUMERIC)),
        ("model", LogisticRegression(max_iter=2000))
    ])
    rf_model = Pipeline([
        ("pre", _build_preprocessor(CLASS_CATEGORICAL, CLASS_NUMERIC)),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    log_pred = log_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    log_acc = accuracy_score(y_test, log_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    best_name = "Random Forest" if rf_acc >= log_acc else "Logistic Regression"
    best_model = rf_model if rf_acc >= log_acc else log_model
    best_pred = rf_pred if rf_acc >= log_acc else log_pred

    preprocessor = best_model.named_steps["pre"]
    feature_names = _get_feature_names(preprocessor, CLASS_CATEGORICAL, CLASS_NUMERIC)

    if best_name == "Random Forest":
        importances = best_model.named_steps["model"].feature_importances_
    else:
        importances = np.abs(best_model.named_steps["model"].coef_[0])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    report = classification_report(y_test, best_pred, output_dict=True)
    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class_or_metric"})
    )

    cm = confusion_matrix(y_test, best_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No/Maybe", "Actual: Yes"],
        columns=["Predicted: No/Maybe", "Predicted: Yes"]
    )

    return {
        "logistic_accuracy": log_acc,
        "rf_accuracy": rf_acc,
        "best_model": best_name,
        "feature_importance": importance_df,
        "classification_report": report_df,
        "confusion_matrix": cm_df
    }

def run_regression(df: pd.DataFrame) -> dict:
    work = df.copy()
    X = work[REG_CATEGORICAL + REG_NUMERIC]
    y = work["Derived_PSM_WTP_Midpoint"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin_model = Pipeline([
        ("pre", _build_preprocessor(REG_CATEGORICAL, REG_NUMERIC)),
        ("model", LinearRegression())
    ])
    rf_model = Pipeline([
        ("pre", _build_preprocessor(REG_CATEGORICAL, REG_NUMERIC)),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    lin_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    lin_pred = lin_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    lin_r2 = r2_score(y_test, lin_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    best_name = "Random Forest Regressor" if rf_r2 >= lin_r2 else "Linear Regression"
    best_model = rf_model if rf_r2 >= lin_r2 else lin_model
    best_pred = rf_pred if rf_r2 >= lin_r2 else lin_pred

    preprocessor = best_model.named_steps["pre"]
    feature_names = _get_feature_names(preprocessor, REG_CATEGORICAL, REG_NUMERIC)

    if best_name == "Random Forest Regressor":
        importances = best_model.named_steps["model"].feature_importances_
    else:
        importances = np.abs(best_model.named_steps["model"].coef_)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    sample_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": best_pred
    }).head(150)

    return {
        "linear_r2": lin_r2,
        "rf_r2": rf_r2,
        "best_model": best_name,
        "feature_importance": importance_df,
        "prediction_sample": sample_df,
        "rmse_best": mean_squared_error(y_test, best_pred) ** 0.5
    }

def run_clustering(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    work = df[["Respondent_ID"] + CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work[CLUSTER_FEATURES])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    work["Cluster"] = [f"Cluster {i+1}" for i in clusters]

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "Respondent_ID": work["Respondent_ID"],
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Cluster": work["Cluster"]
    })

    summary = work.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2).reset_index()

    return {
        "n_clusters": n_clusters,
        "cluster_plot_df": plot_df,
        "cluster_summary": summary
    }
