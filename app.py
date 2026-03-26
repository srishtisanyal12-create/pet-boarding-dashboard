import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.data_prep import load_data, get_overview_metrics, get_feature_rankings, get_concern_rankings
from src.modeling import run_classification, run_regression, run_clustering
from src.association_rules import run_association_rules

st.set_page_config(page_title="Premium Pet Boarding Analytics Dashboard", page_icon="🐾", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "pet_boarding_cleaned.csv"

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

@st.cache_data
def get_classification_results(df):
    return run_classification(df)

@st.cache_data
def get_regression_results(df):
    return run_regression(df)

@st.cache_data
def get_clustering_results(df):
    return run_clustering(df)

@st.cache_data
def get_assoc_results(df):
    return run_association_rules(df)

df = get_data()

st.title("🐾 Premium Pet Boarding App Analytics Dashboard")
st.caption("Business problem: Identify likely adopters, expected willingness to pay, high-value customer segments, and feature bundles for a premium pet boarding app.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Descriptive Analytics",
    "Classification",
    "Regression",
    "Clustering",
    "Association Rules",
    "Recommendations"
])

with tab1:
    st.subheader("Project overview")
    col1, col2, col3, col4 = st.columns(4)
    metrics = get_overview_metrics(df)
    col1.metric("Respondents", f"{metrics['respondents']:,}")
    col2.metric("Variables", f"{metrics['variables']}")
    col3.metric("Likely adopters", f"{metrics['yes_pct']:.1f}%")
    col4.metric("Avg willingness to pay", f"${metrics['avg_wtp']:.0f}")

    st.markdown("""
    **Objective**
    - Predict which pet owners are most likely to adopt the app
    - Estimate how much they are willing to pay
    - Segment the market into actionable personas
    - Identify which features and concerns occur together

    **Dataset**
    - 2,000 synthetic survey responses
    - Cleaned working dataset with 76 analysis-ready columns
    - Includes demographics, pet-care behaviour, feature preferences, trust/concern indices, and willingness-to-pay variables
    """)

    st.info(
        "Use the tabs to explore descriptive patterns, model outputs, customer segments, and feature bundles. "
        "This dashboard is designed for business storytelling rather than technical depth alone."
    )

with tab2:
    st.subheader("Descriptive analytics")

    col1, col2 = st.columns(2)
    adoption_counts = df["Q25_Adoption_Intent"].value_counts().reset_index()
    adoption_counts.columns = ["Adoption_Intent", "Count"]
    fig1 = px.bar(adoption_counts, x="Adoption_Intent", y="Count", title="Adoption intent distribution")
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="Derived_PSM_WTP_Midpoint", nbins=30, title="Willingness to pay distribution")
    col2.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    age_counts = df["Q1_Age_Group"].value_counts().reset_index()
    age_counts.columns = ["Age_Group", "Count"]
    fig3 = px.bar(age_counts, x="Age_Group", y="Count", title="Age group mix")
    col3.plotly_chart(fig3, use_container_width=True)

    pet_counts = df["Q5_Pet_Type"].value_counts().reset_index()
    pet_counts.columns = ["Pet_Type", "Count"]
    fig4 = px.pie(pet_counts, names="Pet_Type", values="Count", title="Pet type mix")
    col4.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)
    feature_df = get_feature_rankings(df)
    fig5 = px.bar(feature_df, x="Average_AddOn_Budget_USD", y="Feature", orientation="h", title="Most valued premium features")
    fig5.update_layout(yaxis={'categoryorder':'total ascending'})
    col5.plotly_chart(fig5, use_container_width=True)

    concern_df = get_concern_rankings(df)
    fig6 = px.bar(concern_df, x="Average_Concern_Score", y="Concern", orientation="h", title="Top customer concerns")
    fig6.update_layout(yaxis={'categoryorder':'total ascending'})
    col6.plotly_chart(fig6, use_container_width=True)

    income_wtp = df.groupby("Q4_Income", as_index=False)["Derived_PSM_WTP_Midpoint"].mean()
    fig7 = px.bar(income_wtp, x="Q4_Income", y="Derived_PSM_WTP_Midpoint", title="Average willingness to pay by income")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    **Quick read**
    - Adoption intent reveals the immediate commercial opportunity.
    - Willingness to pay helps frame pricing and package design.
    - Concern and feature charts tell the product team where trust-building should focus.
    """)

with tab3:
    st.subheader("Classification: who is likely to adopt?")
    classification = get_classification_results(df)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Logistic accuracy", f"{classification['logistic_accuracy']:.3f}")
    metric_cols[1].metric("Random forest accuracy", f"{classification['rf_accuracy']:.3f}")
    metric_cols[2].metric("Selected model", classification["best_model"])

    st.markdown("**Top drivers of adoption**")
    importances = classification["feature_importance"].head(12)
    fig_imp = px.bar(importances, x="importance", y="feature", orientation="h", title="Feature importance")
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.dataframe(classification["classification_report"], use_container_width=True)
    st.success(
        "Business interpretation: adoption is driven most strongly by urgency, trust, concern, and positive attitudes "
        "towards premium care. This means the launch should target owners with real near-term boarding needs and a strong need for transparency."
    )

with tab4:
    st.subheader("Regression: what drives willingness to pay?")
    regression = get_regression_results(df)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Linear Regression R²", f"{regression['linear_r2']:.3f}")
    metric_cols[1].metric("Random Forest R²", f"{regression['rf_r2']:.3f}")
    metric_cols[2].metric("Best model", regression["best_model"])

    st.markdown("**Top drivers of willingness to pay**")
    reg_imp = regression["feature_importance"].head(12)
    fig_reg = px.bar(reg_imp, x="importance", y="feature", orientation="h", title="Drivers of willingness to pay")
    fig_reg.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_reg, use_container_width=True)

    pred_df = regression["prediction_sample"]
    fig_pred = px.scatter(pred_df, x="Actual", y="Predicted", trendline="ols", title="Actual vs predicted willingness to pay")
    st.plotly_chart(fig_pred, use_container_width=True)

    st.info(
        "Higher willingness to pay typically comes from stronger trust, stronger attachment to the pet, "
        "higher current pet spending, and higher-value income bands."
    )

with tab5:
    st.subheader("Clustering: market segments")
    clustering = get_clustering_results(df)

    st.metric("Chosen number of clusters", clustering["n_clusters"])
    fig_cluster = px.scatter(
        clustering["cluster_plot_df"],
        x="PC1", y="PC2", color="Cluster",
        hover_data=["Respondent_ID"],
        title="Cluster map (PCA projection)"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("**Cluster profiles**")
    st.dataframe(clustering["cluster_summary"], use_container_width=True)

    st.markdown("""
    **How to use this**
    - Target premium, trust-sensitive clusters with live updates, CCTV, and emergency coordination.
    - Position budget-sensitive clusters around starter plans or lighter feature bundles.
    - Use segment-level messaging instead of one generic campaign for all pet owners.
    """)

with tab6:
    st.subheader("Association rules: what preferences occur together?")
    assoc = get_assoc_results(df)

    st.metric("Strong rules found", len(assoc["rules"]))
    if len(assoc["rules"]) == 0:
        st.warning("No strong rules were found with the current thresholds.")
    else:
        st.dataframe(assoc["rules"].head(15), use_container_width=True)

        if len(assoc["rules"]) > 0:
            top_rule = assoc["rules"].iloc[0]
            st.success(
                f"Example insight: users who show {top_rule['antecedents']} also tend to show {top_rule['consequents']}. "
                "This helps bundle features and write more relevant campaign messaging."
            )

with tab7:
    st.subheader("Recommendations")
    st.markdown("""
    ### Recommended go-to-market actions

    **1. Target urgent, trust-sensitive owners first**
    - Prioritize users with confirmed or likely near-term need.
    - Use trust-building messages around safety, emergency support, and transparency.

    **2. Lead with a “peace of mind” feature bundle**
    - Package live updates, CCTV access, health logs, and emergency coordination together.
    - These features align with the strongest concern patterns.

    **3. Use segment-based offers**
    - Premium cluster: high-trust, high-attachment users → full-service premium plan
    - Mid-market cluster: selective add-ons → modular package
    - Price-conscious cluster: lower-entry starter plan

    **4. Price with evidence, not guesswork**
    - Use willingness-to-pay outputs to define plan tiers.
    - Offer optional add-ons rather than one high all-inclusive price.

    **5. Build launch messaging around reassurance**
    - The data suggests adoption is not just about convenience.
    - It is about reducing anxiety when owners leave pets in someone else’s care.
    """)
