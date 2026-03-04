import logging
import pandas as pd
import numpy as np
import re

logger = logging.getLogger(__name__)

def clean_dataset(data):
    """
    Cleans a list of dictionaries containing either student or instructor survey data.
    Returns a cleaned list of dictionaries.
    """
    if not data:
        raise ValueError("The provided dataset is empty.")

    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            raise ValueError("The uploaded dataset contains no rows or columns.")

        # Detect survey type
        if 'Total Engage Score-P' in df.columns or 'Content-P_1' in df.columns:
            survey_type = 'instructor'
        else:
            survey_type = 'student'

        if survey_type == 'student':
            # --- Student Cleaning Logic ---
            df = df.iloc[1:].copy()
            df.replace(
                ["nan", "Nan", "NAN", "Unknown", "N/A", "N\\A", ""],
                np.nan,
                inplace=True
            )

            df = df.drop_duplicates()
            df = df.dropna(how='all')

            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
            )

            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({
                    'Yes':'Yes','YES':'Yes','yes':'Yes','Y':'Yes',
                    'No':'No','NO':'No','no':'No','N':'No'
                })

            if "q3" in df.columns and "q3_4_text" in df.columns:
                df["q3"] = df["q3"].where(df["q3"] != "Other", df["q3_4_text"])
                df.drop(["q3_4_text"], axis=1, inplace=True)

            uni_map = {
                "Emu": "Eastern Michigan University",
                "Dha Suffa University": "DHA Suffa University",
                "Dha Suffa University karachi": "DHA Suffa University"
            }
            if 'q2' in df.columns:
                df['q2'] = df['q2'].astype(str).str.strip()
                df['q2'] = df['q2'].replace(uni_map)
                df['q2'] = df['q2'].str.title()
       
            if 'q4' in df.columns:
                df['q4'] = df['q4'].astype(str).str.strip().str.lower()
                df.loc[
                    df['q4'].str.contains("deep", na=False) &
                    df['q4'].str.contains("learn", na=False),
                    'q4'
                ] = "Deep Learning"
                df.loc[df['q4'].str.contains("492", na=False), 'q4'] = "Deep Learning"
                df['q4'] = df['q4'].str.title()

            if 'q6' in df.columns:
                df["q6"] = df["q6"].str.extract(r"(Class instructor|Student)", expand=False)
                df["q6"] = df["q6"].str.strip()

            numeric_columns = ['age', 'rating', 'score']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'start_date' in df.columns:
                df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
                df = df[df['start_date'].notna()]

            if 'email' in df.columns:
                email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                df = df[df['email'].str.match(email_pattern, na=False)]
                df = df.reset_index(drop=True)

            if 'age' in df.columns:
                df.loc[(df['age'] < 10) | (df['age'] > 100), 'age'] = pd.NA

            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna("Unknown")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    median_value = df[col].median()
                    if pd.notna(median_value):
                        df[col] = df[col].fillna(median_value)

        else:
            # --- Instructor Cleaning Logic ---
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("-", "_")
                .str.replace(".", "_", regex=False)
                .str.replace("__", "_", regex=False)
            )

            df = df.iloc[1:].copy()
            df = df.dropna(how='all')
            df = df[df.isnull().mean(axis=1) < 0.8]

            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()
        
            if 'total_engage_score_p' in df.columns:
                df['total_engage_score_p'] = pd.to_numeric(df['total_engage_score_p'], errors='coerce')
                df.loc[(df['total_engage_score_p'] < 0) | (df['total_engage_score_p'] > 100), 'total_engage_score_p'] = pd.NA
        
            if 'startdate' in df.columns:
                df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
                df = df[df['startdate'].notna()]

            df = df.drop_duplicates()
            df = df.reset_index(drop=True)

            if 'q108' in df.columns:
                email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                df['q108'] = df['q108'].astype(str).str.strip().str.lower()
                df.loc[~df['q108'].str.match(email_pattern, na=False), 'q108'] = pd.NA

            if 'q2' in df.columns:
                uni_map = {
                    "Emu": "Eastern Michigan University",
                    "Dha Suffa University": "DHA Suffa University"
                }
                df['q2'] = df['q2'].astype(str).str.strip()
                df['q2'] = df['q2'].replace(uni_map)
                df['q2'] = df['q2'].str.title()
        
            likert_cols = [col for col in df.columns
                if col.startswith((
                    'relevance',
                    'discuss',
                    'act_part',
                    'cls_org',
                    'challenge_level',
                    'cncts'
                ))]

            likert_map = {
                "Never": 1,
                "Rarely": 2,
                "Sometimes": 3,
                "About half the time": 3,
                "Most of the time": 4,
                "Almost always": 5,
                "Always": 5
            }

            for col in likert_cols:
                df[col] = df[col].map(likert_map).astype("float")
        
            for col in likert_cols:
                df.loc[(df[col] < 1) | (df[col] > 5), col] = pd.NA

            if len(likert_cols) > 0:
                df['response_variation'] = df[likert_cols].std(axis=1)
                df = df[df['response_variation'] >= 0.2]
                df = df.drop(columns=['response_variation'])

        import json
        cleaned_data = json.loads(df.to_json(orient='records', date_format='iso'))
        
        if not cleaned_data:
            raise ValueError("All data was filtered out during the cleaning process. Please ensure the data matches the expected format.")
            
        return cleaned_data

    except ValueError as ve:
        # Re-raise intended value errors (like empty dataset)
        raise ve
    except Exception as e:
        logger.error(f"Error during dataset cleaning: {str(e)}", exc_info=True)
        raise ValueError(f"An unexpected error occurred while cleaning the dataset: {str(e)}")