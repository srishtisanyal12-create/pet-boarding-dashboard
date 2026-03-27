import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.data_prep import load_data, get_overview_metrics, get_feature_rankings, get_concern_rankings
from src.modeling import run_classification, run_regression, run_clustering
from src.association_rules import run_association_rules

st.set_page_config(page_title="Premium Pet Boarding Analytics Dashboard", page_icon="🐾", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1240px;
    }
    .stMetric {
        background: rgba(255,255,255,0.03);
        padding: 14px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    div[data-baseweb="tab-list"] {
        gap: 10px;
    }
    div[data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        padding: 10px 15px;
        border-radius: 12px 12px 0 0;
    }
    .insight-box {
        background: linear-gradient(90deg, rgba(16,185,129,0.18), rgba(5,150,105,0.07));
        border-left: 5px solid #10b981;
        padding: 16px;
        border-radius: 12px;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .diag-box {
        background: linear-gradient(90deg, rgba(59,130,246,0.18), rgba(37,99,235,0.07));
        border-left: 5px solid #60a5fa;
        padding: 16px;
        border-radius: 12px;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .warn-box {
        background: linear-gradient(90deg, rgba(245,158,11,0.18), rgba(217,119,6,0.07));
        border-left: 5px solid #f59e0b;
        padding: 16px;
        border-radius: 12px;
        margin-top: 12px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

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
def get_assoc_results(df, min_support, min_confidence, min_lift):
    return run_association_rules(df, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)

df = get_data()

st.title("🐾 Premium Pet Boarding App Analytics Dashboard")
st.markdown("""
<div style='background: linear-gradient(90deg, #1d4ed8, #0f172a); padding: 20px; border-radius: 16px; margin-bottom: 18px;'>
    <h3 style='color: white; margin: 0;'>Trust-led premium pet care, validated through data</h3>
    <p style='color: #dbeafe; margin-top: 8px; font-size: 16px;'>
        This dashboard evaluates market demand, adoption drivers, willingness to pay, customer segments, and feature bundles for a premium pet boarding app.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Overview",
    "Descriptive Analytics",
    "Diagnostic Analytics",
    "Classification",
    "Regression",
    "Clustering",
    "Association Rules",
    "Business Recommendations"
])

# Overview
with tabs[0]:
    st.subheader("Project overview")

    metrics = get_overview_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{metrics['respondents']:,}")
    c2.metric("Variables", f"{metrics['variables']}")
    c3.metric("Likely adopters", f"{metrics['yes_pct']:.1f}%")
    c4.metric("Avg willingness to pay", f"${metrics['avg_wtp']:.0f}")

    st.markdown("""
**Business problem**  
The business wants to identify which pet owners are most likely to adopt a premium pet boarding app, what features they value most, how much they are willing to pay, and how the market can be segmented for targeted offers.

**What this dashboard helps answer**
- Who is most likely to adopt the app?
- What trust and safety concerns are strongest?
- Which features deserve launch priority?
- How much are users likely to pay?
- What customer segments should be targeted differently?
""")

    st.markdown("""
<div class='insight-box'>
<b>Executive takeaway:</b> The strongest opportunity is not just easier booking. It is reduced anxiety through visibility, trust, and premium reassurance.
</div>
""", unsafe_allow_html=True)

# Descriptive
with tabs[1]:
    st.subheader("Descriptive analytics")
    sub1, sub2, sub3 = st.tabs(["Demographics", "Case-specific charts", "Correlations"])

    with sub1:
        c1, c2 = st.columns(2)

        age_counts = df["Q1_Age_Group"].value_counts().reset_index()
        age_counts.columns = ["Age_Group", "Count"]
        fig_age = px.bar(age_counts, x="Age_Group", y="Count", title="Which age groups dominate the potential market?")
        c1.plotly_chart(fig_age, use_container_width=True)
        c1.caption("Business takeaway: the sample is concentrated in active working-age groups, making the concept commercially relevant for digitally comfortable and financially active users.")

        pet_counts = df["Q5_Pet_Type"].value_counts().reset_index()
        pet_counts.columns = ["Pet_Type", "Count"]
        fig_pet = px.pie(pet_counts, names="Pet_Type", values="Count", title="Which pet-owner categories matter most at launch?")
        c2.plotly_chart(fig_pet, use_container_width=True)
        c2.caption("Business takeaway: dogs and cats dominate the addressable market, so launch positioning should prioritise these segments first.")

        income_counts = df["Q4_Income"].value_counts().reset_index()
        income_counts.columns = ["Income", "Count"]
        fig_income = px.bar(income_counts, x="Income", y="Count", title="How is the sample distributed by income?")
        st.plotly_chart(fig_income, use_container_width=True)

    with sub2:
        c1, c2 = st.columns(2)

        adoption_counts = df["Q25_Adoption_Intent"].value_counts().reset_index()
        adoption_counts.columns = ["Adoption_Intent", "Count"]
        fig1 = px.bar(adoption_counts, x="Adoption_Intent", y="Count", title="How strong is initial adoption intent?")
        c1.plotly_chart(fig1, use_container_width=True)
        c1.caption("Business takeaway: the market shows meaningful immediate interest, with 'Yes' and 'Maybe' together creating a sizable launch opportunity.")

        fig2 = px.histogram(df, x="Derived_PSM_WTP_Midpoint", nbins=30, title="How much are users willing to pay?")
        c2.plotly_chart(fig2, use_container_width=True)
        c2.caption("Business takeaway: price tolerance clusters in the middle range, supporting premium positioning with disciplined package design.")

        c3, c4 = st.columns(2)

        feature_df = get_feature_rankings(df)
        fig5 = px.bar(feature_df, x="Average_AddOn_Budget_USD", y="Feature", orientation="h", title="Which premium features matter most?")
        fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
        c3.plotly_chart(fig5, use_container_width=True)
        c3.caption("Business takeaway: users value transparency and control features more than decorative extras.")

        concern_df = get_concern_rankings(df)
        fig6 = px.bar(concern_df, x="Average_Concern_Score", y="Concern", orientation="h", title="What anxieties must the product solve first?")
        fig6.update_layout(yaxis={'categoryorder': 'total ascending'})
        c4.plotly_chart(fig6, use_container_width=True)
        c4.caption("Business takeaway: trust, safety, updates, and emergency handling are the emotional core of the problem.")

        income_wtp = df.groupby("Q4_Income", as_index=False)["Derived_PSM_WTP_Midpoint"].mean()
        fig7 = px.bar(income_wtp, x="Q4_Income", y="Derived_PSM_WTP_Midpoint", title="How does willingness to pay change by income?")
        st.plotly_chart(fig7, use_container_width=True)

    with sub3:
        corr_cols = [
            "Q9_Monthly_Pet_Spend_USD",
            "Derived_Attachment_Index",
            "Derived_Concern_Index",
            "Derived_Trust_Index",
            "Derived_TPB_Composite",
            "Derived_Satisfaction_Index",
            "Derived_PSM_WTP_Midpoint"
        ]
        corr_df = df[corr_cols].corr().round(2)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            text=corr_df.values,
            texttemplate="%{text}",
            colorscale="Blues"
        ))
        fig_corr.update_layout(title="How do the key numeric variables relate to each other?")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Business takeaway: willingness to pay moves alongside pet spend, attachment, trust, and concern rather than existing in isolation.")

# Diagnostic
with tabs[2]:
    st.subheader("Diagnostic analytics: why are these patterns happening?")

    st.markdown("""
<div class='diag-box'>
Descriptive analytics shows <b>what</b> is happening. Diagnostic analytics explains <b>why</b> it is happening.
</div>
""", unsafe_allow_html=True)

    diag1 = (
        df.groupby("Q25_Adoption_Intent")[[
            "Derived_Trust_Index",
            "Derived_Concern_Index",
            "Derived_Attachment_Index"
        ]]
        .mean()
        .reset_index()
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_diag1 = px.bar(
            diag1,
            x="Q25_Adoption_Intent",
            y=["Derived_Trust_Index", "Derived_Concern_Index", "Derived_Attachment_Index"],
            barmode="group",
            title="Why does adoption differ? Trust, concern, and attachment by intent"
        )
        st.plotly_chart(fig_diag1, use_container_width=True)
        st.caption("Business takeaway: likely adopters tend to combine stronger concern, stronger attachment, and stronger trust orientation.")

    adopt_income = (
        df.groupby("Q4_Income")["Q25_Adoption_Intent"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Yes_Adoption_Rate")
    )
    with c2:
        fig_diag2 = px.bar(
            adopt_income,
            x="Q4_Income",
            y="Yes_Adoption_Rate",
            title="Why do some groups adopt more? Yes-adoption rate by income"
        )
        st.plotly_chart(fig_diag2, use_container_width=True)
        st.caption("Business takeaway: affordability matters, but adoption is not limited to one high-income niche.")

    c3, c4 = st.columns(2)
    adopt_freq = (
        df.groupby("Q6_Boarding_Frequency")["Q25_Adoption_Intent"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Yes_Adoption_Rate")
    )
    with c3:
        fig_diag3 = px.bar(
            adopt_freq,
            x="Q6_Boarding_Frequency",
            y="Yes_Adoption_Rate",
            title="Why does urgency matter? Adoption rate by boarding frequency"
        )
        st.plotly_chart(fig_diag3, use_container_width=True)
        st.caption("Business takeaway: users with more frequent boarding needs represent stronger immediate commercial potential.")

    wtp_diag = (
        df.groupby("Q25_Adoption_Intent")["Derived_PSM_WTP_Midpoint"]
        .mean()
        .reset_index()
    )
    with c4:
        fig_diag4 = px.bar(
            wtp_diag,
            x="Q25_Adoption_Intent",
            y="Derived_PSM_WTP_Midpoint",
            title="Why is monetization viable? Willingness to pay by adoption intent"
        )
        st.plotly_chart(fig_diag4, use_container_width=True)
        st.caption("Business takeaway: the users most likely to adopt also tend to show stronger willingness to pay.")

    st.markdown("""
<div class='insight-box'>
<b>Diagnostic insight:</b> The strongest explanation for adoption is not convenience alone. It is the combination of emotional attachment, worry about care quality, and desire for visible reassurance.
</div>
""", unsafe_allow_html=True)

# Classification
with tabs[3]:
    st.subheader("Classification: who is likely to adopt?")
    classification = get_classification_results(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic accuracy", f"{classification['logistic_accuracy']:.3f}")
    c2.metric("Random forest accuracy", f"{classification['rf_accuracy']:.3f}")
    c3.metric("Selected model", classification["best_model"])

    left, right = st.columns(2)

    with left:
        st.markdown("**Top drivers of adoption**")
        imp = classification["feature_importance"].head(12)
        fig_imp = px.bar(imp, x="importance", y="feature", orientation="h", title="Which variables drive likely adoption?")
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    with right:
        st.markdown("**Confusion matrix**")
        if "confusion_matrix" in classification:
            cm = classification["confusion_matrix"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm.values,
                x=cm.columns,
                y=cm.index,
                text=cm.values,
                texttemplate="%{text}",
                colorscale="Teal"
            ))
            fig_cm.update_layout(title="Where is the model right and where does it get confused?")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix not available. Please update src/modeling.py and reboot the app.")

    st.dataframe(classification["classification_report"], use_container_width=True)

    st.markdown("""
<div class='insight-box'>
<b>Business takeaway:</b> adoption is strongest among users with stronger concern, stronger trust orientation, higher pet spend, and more positive attitudes toward premium monitored care.
</div>
""", unsafe_allow_html=True)

# Regression
with tabs[4]:
    st.subheader("Regression: what drives willingness to pay?")
    regression = get_regression_results(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linear Regression R²", f"{regression['linear_r2']:.3f}")
    c2.metric("Random Forest R²", f"{regression['rf_r2']:.3f}")
    c3.metric("Best model", regression["best_model"])
    c4.metric("Best RMSE", f"{regression['rmse_best']:.2f}")

    left, right = st.columns(2)

    with left:
        reg_imp = regression["feature_importance"].head(12)
        fig_reg = px.bar(reg_imp, x="importance", y="feature", orientation="h", title="What most strongly influences willingness to pay?")
        fig_reg.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_reg, use_container_width=True)

    with right:
        pred_df = regression["prediction_sample"]
        fig_pred = px.scatter(pred_df, x="Actual", y="Predicted", title="How close are predicted and actual willingness to pay values?")
        st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("""
<div class='insight-box'>
<b>Business takeaway:</b> willingness to pay is shaped not only by income, but by emotional attachment, current pet spending, concern intensity, and the perceived value of reassurance features.
</div>
""", unsafe_allow_html=True)

# Clustering
with tabs[5]:
    st.subheader("Clustering: market segments")
    clustering = get_clustering_results(df)

    st.metric("Chosen number of clusters", clustering["n_clusters"])

    fig_cluster = px.scatter(
        clustering["cluster_plot_df"],
        x="PC1", y="PC2", color="Cluster",
        hover_data=["Respondent_ID"],
        title="How do respondents cluster into different customer personas?"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("**Cluster profiles**")
    st.dataframe(clustering["cluster_summary"], use_container_width=True)

    st.markdown("""
<div class='insight-box'>
<b>Business takeaway:</b> different customer groups combine attachment, concern, trust, spend, and willingness to pay in different ways, so one generic campaign would be inefficient.
</div>
""", unsafe_allow_html=True)

# Association Rules
with tabs[6]:
    st.subheader("Association rules: what preferences occur together?")

    st.markdown("""
<div class='warn-box'>
You can change the rule strictness below. Higher values make the rules fewer but stronger.
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    min_support = c1.slider("Minimum support", 0.01, 0.30, 0.08, 0.01)
    min_confidence = c2.slider("Minimum confidence", 0.05, 0.90, 0.25, 0.05)
    min_lift = c3.slider("Minimum lift", 0.50, 3.00, 1.00, 0.10)

    assoc = get_assoc_results(df, min_support, min_confidence, min_lift)

    st.metric("Strong rules found", len(assoc["rules"]))

    if len(assoc["rules"]) == 0:
        st.warning("No strong rules were found with the current thresholds. Try lowering the sliders slightly.")
    else:
        st.dataframe(assoc["rules"].head(20), use_container_width=True)
        top_rule = assoc["rules"].iloc[0]
        st.markdown(f"""
<div class='insight-box'>
<b>Top rule insight:</b> users showing <b>{top_rule['antecedents']}</b> also tend to show <b>{top_rule['consequents']}</b>. This helps bundle features and write more relevant campaign messaging.
</div>
""", unsafe_allow_html=True)

# Recommendations
with tabs[7]:
    st.subheader("Business recommendations")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
### 1. Launch around reassurance, not convenience
- Lead with live updates, CCTV, emergency coordination, and verified caretaker profiles
- Position the product as a trust-building platform, not just a booking tool

### 2. Prioritise high-intent segments first
- Target urgent and trust-sensitive owners
- Focus on users with stronger pet attachment and meaningful current pet spend

### 3. Use tiered pricing
- Full premium plan for high-trust / high-value users
- Modular plan for mid-premium users
- Starter plan for price-sensitive users
""")

    with c2:
        st.markdown("""
### 4. Bundle features based on actual co-occurrence
- Package updates, CCTV, health logs, and emergency coordination together
- Avoid treating these as isolated add-ons

### 5. Build segment-specific messaging
- Premium cluster: peace of mind and concierge-like care
- Mid segment: monitored care with selective add-ons
- Budget segment: safe, simpler entry-level plan

### 6. Use the dashboard as a decision tool
- Descriptive = what is happening
- Diagnostic = why it is happening
- Classification = who to target
- Regression = how to price
- Clustering = how to segment
- Association rules = how to bundle
""")

    st.markdown("""
<div class='insight-box'>
<b>Final business insight:</b> this idea wins when it reduces the emotional risk of leaving a pet behind. The strongest commercial lever is trust-backed transparency.
</div>
""", unsafe_allow_html=True)
