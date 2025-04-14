import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\class work\HVAC_Dynamic_Fuzzy_PID_2017_with_Target.csv")
    df.dropna(inplace=True)
    return df

# Train model
@st.cache_resource
def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model, X.columns

# App
def main():
    st.title("⚡ Energy Consumption Optimizer")

    df = load_data()

    # Identify target column
    target_col = "Target"
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
        return

    model, feature_cols = train_model(df, target_col)

    st.subheader("🔢 Enter Input Values:")

    user_input = {}
    for col in feature_cols:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        user_input[col] = val

    if st.button("🚀 Predict Energy Consumption"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Energy Consumption: {prediction:.2f}")

if __name__ == "__main__":
    main()
