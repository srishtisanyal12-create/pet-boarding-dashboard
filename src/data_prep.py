from pathlib import Path
import pandas as pd

FEATURE_LABELS = {
    "Q14_Budget_Live_Video": "Live video updates",
    "Q14_Budget_Verified_Profiles": "Verified caretaker profiles",
    "Q14_Budget_Messaging": "In-app messaging",
    "Q14_Budget_CCTV": "CCTV access",
    "Q14_Budget_Custom_Care": "Custom care plans",
    "Q14_Budget_Health_Log": "Health log updates",
    "Q14_Budget_Emergency_Coord": "Emergency coordination",
    "Q14_Budget_Grooming": "Grooming add-on",
}

CONCERN_LABELS = {
    "Q13_Concern_Safety": "Safety",
    "Q13_Concern_Feeding": "Feeding",
    "Q13_Concern_No_Updates": "No updates",
    "Q13_Concern_Caretaker_Exp": "Caretaker experience",
    "Q13_Concern_Hygiene": "Hygiene",
    "Q13_Concern_Pet_Emotion": "Pet emotional wellbeing",
    "Q13_Concern_Emergency": "Emergency handling",
    "Q13_Concern_Trust": "Trust",
}

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def get_overview_metrics(df: pd.DataFrame) -> dict:
    yes_pct = (df["Q25_Adoption_Intent"].eq("Yes").mean()) * 100
    return {
        "respondents": len(df),
        "variables": df.shape[1],
        "yes_pct": yes_pct,
        "avg_wtp": df["Derived_PSM_WTP_Midpoint"].mean(),
    }

def get_feature_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in FEATURE_LABELS.items():
        rows.append({"Feature": label, "Average_AddOn_Budget_USD": df[col].mean()})
    return pd.DataFrame(rows).sort_values("Average_AddOn_Budget_USD", ascending=False)

def get_concern_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in CONCERN_LABELS.items():
        rows.append({"Concern": label, "Average_Concern_Score": df[col].mean()})
    return pd.DataFrame(rows).sort_values("Average_Concern_Score", ascending=False)
