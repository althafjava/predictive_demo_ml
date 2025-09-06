import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

st.title("Predictive Maintenance Dashboard")

# -----------------------------
# 1. Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload Monthly Machine Data CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("CSV Loaded Successfully!")
    
    # Save uploaded CSV locally
    os.makedirs("data", exist_ok=True)
    data.to_csv("data/machine_data.csv", index=False)
else:
    # Load existing CSV if no upload
    data = pd.read_csv("data/machine_data.csv")

# -----------------------------
# 2. Retrain model if requested
# -----------------------------
retrain = st.checkbox("Retrain model with current data", value=False)

if retrain:
    X = data[['shot_count', 'major_repair', 'minor_repair']]
    y = data['breakdowns']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model retrained! Test Accuracy: {acc:.2f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/maintenance_model.pkl")
else:
    # Load existing model
    model = joblib.load("models/maintenance_model.pkl")

# -----------------------------
# 3. Show historical data
# -----------------------------
st.subheader("Historical Machine Data")
st.dataframe(data)

# -----------------------------
# 4. Select machine
# -----------------------------
machine = st.selectbox("Select Machine", data['machine_id'].unique())
machine_data = data[data['machine_id'] == machine]

st.subheader(f"Data for {machine}")
st.dataframe(machine_data)

# -----------------------------
# 5. Predict breakdown probability
# -----------------------------
latest_features = machine_data.iloc[-1][['shot_count', 'major_repair', 'minor_repair']].values.reshape(1, -1)
pred_prob = model.predict_proba(latest_features)[0][1]
st.subheader("Predicted Breakdown Probability for Next Month")
st.metric(label="Breakdown Probability", value=f"{pred_prob*100:.2f}%")

# -----------------------------
# 6. Visualizations
# -----------------------------
st.subheader("Shot Count Over Time")
fig1 = px.line(machine_data, x='month', y='shot_count', title=f"{machine} Shot Count Trend")
st.plotly_chart(fig1)

st.subheader("Breakdowns Over Time")
fig2 = px.bar(machine_data, x='month', y='breakdowns', title=f"{machine} Breakdowns Trend")
st.plotly_chart(fig2)

# -----------------------------
# 7. Predict for all machines (optional)
# -----------------------------
if st.checkbox("Show breakdown predictions for all machines"):
    all_preds = []
    for m_id in data['machine_id'].unique():
        m_data = data[data['machine_id'] == m_id]
        features = m_data.iloc[-1][['shot_count', 'major_repair', 'minor_repair']].values.reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        all_preds.append({"machine_id": m_id, "breakdown_prob": prob*100})
    st.subheader("Predictions for All Machines")
    st.dataframe(pd.DataFrame(all_preds))

