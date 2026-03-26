# Premium Pet Boarding App Analytics Dashboard

This project turns a synthetic survey dataset into a business analytics dashboard for a premium pet boarding app with live updates and personalised care.

## Business problem
The company wants to identify which pet owners are most likely to adopt the app, what features they value most, how much they are willing to pay, and what customer segments exist so it can create targeted packages, improve adoption, and maximise revenue.

## Project contents
- `data/pet_boarding_cleaned.csv` — cleaned analysis-ready dataset
- `app.py` — Streamlit dashboard
- `src/data_prep.py` — helper functions for loading data and descriptive summaries
- `src/modeling.py` — classification, regression, and clustering logic
- `src/association_rules.py` — association rule mining logic
- `src/eda.py` — simple EDA export script
- `requirements.txt` — Python dependencies

## Cleaning and validation summary
The original workbook was already largely structured. A light validation pass found:
- no duplicate respondent IDs
- no fully duplicated rows
- one typo-like column with almost all values missing: `Q10_Attention_5`
- a small amount of synthetic logical noise in a few multi-select usage responses

To preserve business meaning and avoid over-cleaning, the project:
- dropped `Q10_Attention_5`
- retained the rest of the dataset intact
- used the cleaned version as the main CSV for analysis

## Analytical methods used
### 1. Classification
Target: likely app adoption  
Variable used: `Q25_Adoption_Intent` (converted to binary yes vs not-yes)

### 2. Regression
Target: willingness to pay  
Variable used: `Derived_PSM_WTP_Midpoint`

### 3. Clustering
Goal: identify customer personas using trust, concern, attachment, satisfaction, and spending indicators

### 4. Association Rule Mining
Goal: identify which concerns and desired features occur together

## How to run locally
1. Create a project folder and place all files in the structure shown below.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the dashboard:

```bash
streamlit run app.py
```

## Recommended folder structure
```text
pet_boarding_streamlit_project/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── pet_boarding_cleaned.csv
└── src/
    ├── data_prep.py
    ├── modeling.py
    ├── association_rules.py
    └── eda.py
```

## GitHub upload steps
1. Create a new GitHub repository.
2. Upload all files and folders exactly as shown above.
3. Make sure `app.py` is in the repository root.
4. Make sure `requirements.txt` is also in the repository root.

## Streamlit deployment steps
1. Go to Streamlit Community Cloud.
2. Sign in with GitHub.
3. Choose **New app**.
4. Select your repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

## Common deployment issues
- Missing library in `requirements.txt`
- Wrong file path for the dataset
- `app.py` not saved in the root folder
- CSV missing from the `data/` folder

## Suggested presentation angle
When you present this dashboard, focus on:
- demand validation
- who to target first
- which features to bundle
- what price tiers appear feasible
- why a trust-led launch strategy matters
