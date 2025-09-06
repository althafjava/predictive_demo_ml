import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# -----------------------------
# 1. Load the CSV data
# -----------------------------
data = pd.read_csv("data/machine_data.csv")

# -----------------------------
# 2. Prepare features and target
# -----------------------------
X = data[['shot_count', 'major_repair', 'minor_repair']]  # input features
y = data['breakdowns']                                   # target variable

# -----------------------------
# 3. Split into train and test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Random Forest Classifier
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.2f}")

# -----------------------------
# 6. Save trained model
# -----------------------------
os.makedirs("models", exist_ok=True)  # make sure folder exists
joblib.dump(model, "models/maintenance_model.pkl")
print("Model saved to models/maintenance_model.pkl")

