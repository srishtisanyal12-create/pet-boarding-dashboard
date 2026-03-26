import pandas as pd

ASSOCIATION_SOURCE = {
    "High concern: safety": "Q13_Concern_Safety",
    "High concern: no updates": "Q13_Concern_No_Updates",
    "High concern: emergency": "Q13_Concern_Emergency",
    "High concern: trust": "Q13_Concern_Trust",
    "Needs live video": "Q14_Budget_Live_Video",
    "Needs CCTV": "Q14_Budget_CCTV",
    "Needs health log": "Q14_Budget_Health_Log",
    "Needs emergency coordination": "Q14_Budget_Emergency_Coord",
    "Needs custom care": "Q14_Budget_Custom_Care",
    "Urgent or likely need": "Q11_Urgency",
}

def run_association_rules(df: pd.DataFrame, min_support: float = 0.08, min_confidence: float = 0.25, min_lift: float = 1.0) -> dict:
    basket = pd.DataFrame(index=df.index)

    for label, col in ASSOCIATION_SOURCE.items():
        if col == "Q11_Urgency":
            basket[label] = df[col].isin(["Yes - confirmed", "Maybe - possible"])
        elif "concern" in label.lower():
            basket[label] = df[col] >= 5
        else:
            basket[label] = df[col] > 0

    cols = basket.columns.tolist()
    rows = []

    for a in cols:
        a_support = basket[a].mean()
        if a_support == 0:
            continue
        for b in cols:
            if a == b:
                continue
            both = (basket[a] & basket[b]).mean()
            if both < min_support:
                continue
            confidence = both / a_support if a_support else 0
            b_support = basket[b].mean()
            lift = confidence / b_support if b_support else 0
            if confidence >= min_confidence and lift >= min_lift:
                rows.append({
                    "antecedents": a,
                    "consequents": b,
                    "support": round(float(both), 3),
                    "confidence": round(float(confidence), 3),
                    "lift": round(float(lift), 3),
                })

    rules = pd.DataFrame(rows).sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift"])
    return {"rules": rules}
