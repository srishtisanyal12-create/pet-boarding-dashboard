from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "pet_boarding_cleaned.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    summary = df.describe(include="all").transpose()
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)

    adoption = df["Q25_Adoption_Intent"].value_counts().reset_index()
    adoption.columns = ["Adoption_Intent", "Count"]
    adoption.to_csv(OUTPUT_DIR / "adoption_counts.csv", index=False)

    print("EDA outputs saved in outputs/")

if __name__ == "__main__":
    main()
