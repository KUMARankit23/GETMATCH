# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import streamlit as st
import os

DATA_FILE = "Get Matched to Your Ideal College Club! (Responses) - Form responses 1 (2) (3).csv"
MODEL_FILE = "club_predictor.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES_FILE = "feature_columns.pkl"
ENCODER_FILE = "label_encoder.pkl"

if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEATURES_FILE) and os.path.exists(ENCODER_FILE)):
    # 1. Load the dataset
    df = pd.read_csv(DATA_FILE)

    # 2. Clean column names
    df.columns = [
        "timestamp", "email", "name", "year", "interests", "personality_traits",
        "motivation", "prior_club_participation", "clubs_joined", "event_count_last_semester",
        "preferred_event_type", "availability", "open_to_other_clubs"
    ]
    df.columns = [col.strip().lower() for col in df.columns]

    # 3. Normalize string-type values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # 4. Replace empty strings and known placeholders with NaN
    df.replace(['', 'nan', 'none', 'n/a'], pd.NA, inplace=True)

    # 5. Handle multi-select fields using MultiLabelBinarizer
    def split_and_clean(col, delimiter=','):
        return df[col].dropna().apply(lambda x: [i.strip() for i in x.split(delimiter)])

    mlb_interests = MultiLabelBinarizer()
    mlb_traits = MultiLabelBinarizer()
    mlb_motivation = MultiLabelBinarizer()

    df_interests = pd.DataFrame(
        mlb_interests.fit_transform(split_and_clean("interests")),
        columns=["interest_" + i for i in mlb_interests.classes_]
    )
    df_traits = pd.DataFrame(
        mlb_traits.fit_transform(split_and_clean("personality_traits")),
        columns=["trait_" + i for i in mlb_traits.classes_]
    )
    df_motivation = pd.DataFrame(
        mlb_motivation.fit_transform(split_and_clean("motivation")),
        columns=["motivation_" + i for i in mlb_motivation.classes_]
    )

    # 6. Reset index before merging
    df = df.reset_index(drop=True)
    df_interests = df_interests.reset_index(drop=True)
    df_traits = df_traits.reset_index(drop=True)
    df_motivation = df_motivation.reset_index(drop=True)

    # 7. Merge all encoded features
    df_processed = pd.concat([df, df_interests, df_traits, df_motivation], axis=1)

    # 8. Drop original multi-select text fields
    df_processed.drop(columns=["interests", "personality_traits", "motivation"], inplace=True)

    # 9. Encode all categorical columns (object type) to numeric
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna('missing')
        df_processed[col] = LabelEncoder().fit_transform(df_processed[col].astype(str))

    # 10. Define target column
    target_column = 'clubs_joined'  # Use the actual club joined as the target
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' not found in processed data columns: {df_processed.columns}")

    # 11. Save label encoder for the target
    le = LabelEncoder()
    df_processed[target_column] = le.fit_transform(df_processed[target_column].astype(str))
    joblib.dump(le, ENCODER_FILE)  # Save the encoder for later use

    # 12. Drop non-informative columns (e.g., timestamp, email, name)
    X = df_processed.drop(columns=[target_column, 'timestamp', 'email', 'name'])
    y = df_processed[target_column]

    # 13. Remove rows with NaN in features or target
    mask = ~X.isnull().any(axis=1) & ~y.isnull()
    X = X[mask]
    y = y[mask]

    # 14. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 15. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 16. Train Random Forest Model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)

    # 17. Predictions and Evaluation
    y_pred = rf.predict(X_test_scaled)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.2f}")

    # 18. Save model, scaler, and feature columns
    joblib.dump(rf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(list(X.columns), FEATURES_FILE)
    print("✅ Model trained and saved!")


# Load model and encoders
model = joblib.load("club_predictor.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
le = joblib.load("label_encoder.pkl")

# --- Sidebar ---
st.sidebar.image("club_images/campus.png", width=180)
st.sidebar.title("About")
st.sidebar.info(
    "Welcome to the BMU Club/Society Recommender! "
    "Select your interests and personality traits to discover the best club for you. "
    "Enjoy exploring our vibrant campus life!"
)

# --- Main Title with Campus Image ---
st.image("club_images/campus.png", use_column_width=True)
st.markdown(
    "<h1 style='text-align: center; color: #FFD700;'>🎓 Club/Society Recommender for New Students</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Interests and Traits ---
interest_columns = [col for col in feature_columns if col.startswith("interest_")]
trait_columns = [col for col in feature_columns if col.startswith("trait_")]

interest_options = [col.replace("interest_", "") for col in interest_columns]
trait_options = [col.replace("trait_", "") for col in trait_columns]

# Use columns for a modern look
col1, col2 = st.columns(2)
with col1:
    selected_interests = st.multiselect(
        "Select your interests:",
        options=interest_options,
        format_func=lambda x: x.replace("_", " ").title()
    )
with col2:
    selected_traits = st.multiselect(
        "Select your personality traits:",
        options=trait_options,
        format_func=lambda x: x.replace("_", " ").title()
    )

st.markdown("---")

if st.button("🎯 Predict Best Club"):
    if not selected_interests or not selected_traits:
        st.warning("Please select both interests and personality traits.")
    else:
        try:
            user_input = []
            for col in feature_columns:
                if col.startswith("interest_"):
                    key = col.replace("interest_", "")
                    user_input.append(1 if key in selected_interests else 0)
                elif col.startswith("trait_"):
                    key = col.replace("trait_", "")
                    user_input.append(1 if key in selected_traits else 0)
                else:
                    user_input.append(0)
            X_scaled = scaler.transform([user_input])
            prediction = model.predict(X_scaled)
            club_name = le.inverse_transform([int(prediction[0])])[0]
            club_image_map = {
                "adventure club": "club-logo-black-Transparent-1-Adventure-Club.png",
                "photography club": "photography_club.png",
                "theatre society": "logo-full-Kavya-Sekhar.png",
                # Add more mappings as needed
            }
            st.success(f"🎯 Based on your profile, you should explore the **{club_name.title()}** club!")
            image_file = club_image_map.get(club_name.lower(), None)
            image_path = f"club_images/{image_file}" if image_file else None
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption=f"{club_name.title()} Club", use_column_width=True)
            else:
                st.image("club_images/default.png", caption="No image available for this club.", use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>Made with ❤️ at BMU</div>",
    unsafe_allow_html=True
)